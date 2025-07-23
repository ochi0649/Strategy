from typing import Optional, Tuple
from lib.interact.tile import Tile
from lib.interface.queries.typing import QueryType
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.events.moves.move_place_meeple import MovePlaceMeeple, MovePlaceMeeplePass
from lib.interface.events.moves.typing import MoveType
from lib.config.map_config import MAX_MAP_LENGTH, MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType
from helper.game import Game

IMMEDIATE_WEIGHT = 6.0
FUTURE_WEIGHT    = 0.2
BLOCK_WEIGHT     = 0.4
MERGE_WEIGHT     = 0.8
MEEPLE_PENALTY   = 3

NEIGHBOR_OFFSETS = [((0,-1),'top_edge'), ((1,0),'right_edge'), ((0,1),'bottom_edge'), ((-1,0),'left_edge')]
OPPOSITE = {'top_edge':'bottom_edge','bottom_edge':'top_edge','left_edge':'right_edge','right_edge':'left_edge'}
OFFSETS1 = {'top_edge':(0,-1), 'right_edge':(1,0), 'bottom_edge':(0,1), 'left_edge':(-1,0)}
OFFSETS2 = {'top_edge':(0,-2), 'right_edge':(2,0), 'bottom_edge':(0,2), 'left_edge':(-2,0)}

class BotState:
    def __init__(self):
        self.last_tile: Optional[Tile] = None

def main():
    game = Game()
    bot_state = BotState()
    while True:
        query = game.get_next_query()
        match query:
            case QueryPlaceTile():
                move = handle_place_tile(game, bot_state, query)
            case QueryPlaceMeeple():
                move = handle_place_meeple(game, bot_state, query)
            case _:
                raise RuntimeError('Unexpected query')
        game.send_move(move)

def river_is_valid(grid: list[list[Tile|None]], tile: Tile, nx: int, ny: int) -> bool:
    river_flag = False
    river_connections = 0
    for (dx, dy), edge in NEIGHBOR_OFFSETS:
        st = tile.internal_edges[edge]
        if st == StructureType.RIVER:
            river_flag = True
            n = None
            x2, y2 = nx+dx, ny+dy
            if 0 <= x2 < MAX_MAP_LENGTH and 0 <= y2 < MAX_MAP_LENGTH:
                n = grid[y2][x2]
            if n:
                river_connections += 1
                if river_connections > 1:
                    return False
            else:
                fx, fy = nx+OFFSETS1[edge][0], ny+OFFSETS1[edge][1]
                for odx, ody in OFFSETS1.values():
                    cx, cy = fx+odx, fy+ody
                    if (cx, cy) != (nx, ny) and 0 <= cx < MAX_MAP_LENGTH and 0 <= cy < MAX_MAP_LENGTH:
                        if grid[cy][cx]:
                            return False
                fx2, fy2 = nx+OFFSETS2[edge][0], ny+OFFSETS2[edge][1]
                for odx, ody in OFFSETS1.values():
                    cx, cy = fx2+odx, fy2+ody
                    if 0 <= cx < MAX_MAP_LENGTH and 0 <= cy < MAX_MAP_LENGTH:
                        if grid[cy][cx]:
                            return False
    if river_flag and river_connections == 0:
        return False
    return True

def handle_place_tile(game: Game, bot_state: BotState, q: QueryPlaceTile) -> MovePlaceTile:
    grid = game.state.map._grid
    spots: list[Tuple[int,int]] = []
    seen = set()
    for placed in game.state.map.placed_tiles:
        px, py = placed.placed_pos
        for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:
            nx, ny = px+dx, py+dy
            if 0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH and grid[ny][nx] is None and (nx,ny) not in seen:
                for ddx, ddy in [(0,-1),(1,0),(0,1),(-1,0)]:
                    adj_x, adj_y = nx+ddx, ny+ddy
                    if 0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH:
                        if grid[adj_y][adj_x] is not None:
                            seen.add((nx,ny))
                            spots.append((nx,ny))
                            break

    best_score = float('-inf')
    best_idx = best_rot = 0
    best_x = best_y = 0

    for idx, tile in enumerate(game.state.my_tiles):
        orig_rot = tile.rotation
        for r in range(4):
            if r > 0:
                tile.rotate_clockwise(1)
            for nx, ny in spots:
                if not game.can_place_tile_at(tile, nx, ny):
                    continue
                if not river_is_valid(grid, tile, nx, ny):
                    continue
                tile.placed_pos = (nx, ny)
                imm = estimate_immediate_score(game, tile)
                fut = estimate_future_score(game, tile)
                blk = estimate_blocking(game, tile)
                mrg = estimate_merge(game, tile)
                score = (IMMEDIATE_WEIGHT*imm + FUTURE_WEIGHT*fut + BLOCK_WEIGHT*blk + MERGE_WEIGHT*mrg)
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_rot = tile.rotation
                    best_x, best_y = nx, ny
        while tile.rotation != orig_rot:
            tile.rotate_clockwise(1)

    chosen = game.state.my_tiles[best_idx]
    delta = (best_rot - chosen.rotation) % 4
    for _ in range(delta): chosen.rotate_clockwise(1)

    neighbour_exists = any(
        0 <= best_x + dx < MAX_MAP_LENGTH and 0 <= best_y + dy < MAX_MAP_LENGTH and grid[best_y + dy][best_x + dx]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
    )
    if not neighbour_exists:
        raise RuntimeError(f"Invalid placement: no neighbours at ({best_x}, {best_y})")

    chosen.placed_pos = (best_x, best_y)
    model = chosen._to_model()
    bot_state.last_tile = chosen
    return game.move_place_tile(q, model, best_idx)

def handle_place_meeple(game: Game, bot_state: BotState, q: QueryPlaceMeeple):
    recent = bot_state.last_tile
    if not recent or game.state.me.num_meeples == 0:
        return game.move_place_meeple_pass(q)
    meeples_left = game.state.me.num_meeples
    model = recent._to_model()
    placeable = game.state.get_placeable_structures(model)
    completed = game.state.get_completed_components(recent)

    def unclaimed(t: Tile, edge: str) -> bool:
        return len(game.state._get_claims(t, edge)) == 0

    def expected(edge: str) -> float:
        return game.state._get_reward(recent, edge, partial=True)

    if MONASTARY_IDENTIFIER in placeable:
        return game.move_place_meeple(q, model, MONASTARY_IDENTIFIER)

    best_edge, best_score = None, -1.0
    for edge, st in placeable.items():
        if edge in completed or not unclaimed(recent, edge): continue
        val = expected(edge)
        if meeples_left == 1: val -= MEEPLE_PENALTY
        if val > best_score: best_score, best_edge = val, edge

    if best_edge:
        if best_score >= 4: return game.move_place_meeple(q, model, best_edge)
        if meeples_left <= 2 and best_score >= 3: return game.move_place_meeple(q, model, best_edge)
        if meeples_left > 2 and best_score >= 2: return game.move_place_meeple(q, model, best_edge)

    for edge, st in placeable.items():
        if edge not in completed and unclaimed(recent, edge):
            return game.move_place_meeple(q, model, edge)

    return game.move_place_meeple_pass(q)

def estimate_immediate_score(game: Game, tile: Tile) -> float:
    return sum(game.state._get_reward(tile, e) for e in game.state.get_completed_components(tile))

def estimate_future_score(game: Game, tile: Tile) -> float:
    return sum(game.state._get_reward(tile, edge, partial=True)*0.7 for edge, st in tile.internal_edges.items() if StructureType.can_claim(st))

def estimate_blocking(game: Game, tile: Tile) -> float:
    score = 0.0
    me = game.state.me.player_id
    x, y = tile.placed_pos
    for (dx, dy), edge in NEIGHBOR_OFFSETS:
        nx, ny = x+dx, y+dy
        if 0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH:
            nb = game.state.map._grid[ny][nx]
            if nb:
                opp = OPPOSITE[edge]
                for o in game.state._get_claims(nb, opp):
                    if o != me:
                        score += game.state._get_reward(nb, opp)*0.1
    return score

def estimate_merge(game: Game, tile: Tile) -> float:
    score = 0.0
    me = game.state.me.player_id
    x, y = tile.placed_pos
    for (dx, dy), edge in NEIGHBOR_OFFSETS:
        nx, ny = x+dx, y+dy
        if 0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH:
            nb = game.state.map._grid[ny][nx]
            if nb and tile.internal_edges[edge] == nb.internal_edges.get(OPPOSITE[edge]) and StructureType.can_claim(tile.internal_edges[edge]):
                owners = game.state._get_claims(nb, OPPOSITE[edge])
                if owners:
                    cnt = {p: owners.count(p) for p in set(owners)}
                    if cnt.get(me, 0) >= max(cnt.values()):
                        score += game.state._get_reward(tile, edge)
                    else:
                        score -= game.state._get_reward(tile, edge)*0.5
    return score

if __name__ == '__main__':
    main()