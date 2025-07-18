from re import sub
from typing import cast
from lib.interface.events.typing import EventPlayerWon
from engine.config.game_config import (
    MAX_ROUNDS,
    NUM_TILES_DRAWN_PER_ROUND,
    NUM_PLAYERS,
    NUM_TILES_IN_HAND,
)
from engine.interface.io.censor_event import CensorEvent
from engine.interface.io.exceptions import PlayerException
from engine.interface.io.game_result import (
    GameBanResult,
    GameCancelledResult,
    GameSuccessResult,
)
from engine.interface.io.input_validator import MoveValidator
from engine.interface.logging.event_factory import event_banned_factory
from engine.interface.logging.event_inspector import EventInspector
from engine.state.game_state import GameState
from engine.config.io_config import CORE_DIRECTORY
from engine.game.tile_subscriber import MonastaryNeighbourSubsciber

from engine.state.player_state import PlayerState
from engine.state.state_mutator import StateMutator

from lib.config.expansion import EXPANSION
from lib.config.map_config import MAP_CENTER, TILE_EDGE_IDS, TILE_EXTERNAL_POS
from lib.interact.structure import StructureType
from lib.interact.tile import MONASTARY_IDENTIFIER, NO_POINTS, Tile
from lib.interface.events.event_game_ended import (
    EventGameEndedStaleMate,
)
from lib.interface.events.event_game_started import EventGameStarted
from lib.interface.events.event_player_drew_tiles import EventPlayerDrewTiles
from lib.interface.events.event_player_meeple_freed import EventPlayerMeepleFreed
from lib.interface.events.event_river_phase_completed import EventRiverPhaseCompleted
from lib.interface.events.event_tile_placed import EventStartingTilePlaced

from random import sample
import shutil


class GameEngine:
    def __init__(self, print_recording_interactive: bool = False) -> None:
        print("Intialising game engine!")

        self.state = GameState()
        self.validator = MoveValidator(self.state)
        self.mutator = StateMutator(self.state)
        self.censor = CensorEvent(self.state)

    def start(self) -> None:
        try:
            self.state._connect_players()
            self.run_game()
        except PlayerException as e:
            event = event_banned_factory(e)
            self.mutator.commit(event)
        finally:
            self.finish()

    def run_game(self) -> None:
        assert NUM_PLAYERS == len(self.state.players)
        turn_order = sample(list(self.state.players.keys()), k=NUM_PLAYERS)
        self.state.turn_order = turn_order

        while not self.state.is_game_over():
            print(f"New round {self.state.round + 1}", flush=True)

            if self.state.round == -1:
                self.mutator.commit(
                    EventGameStarted(
                        turn_order=self.state.turn_order,
                        players=[
                            player._to_player_model()
                            for player in self.state.players.values()
                        ],
                    )
                )

                self.state.start_river_phase()
                self.state.map.place_river_start(MAP_CENTER)
                self.mutator.commit(
                    EventStartingTilePlaced(
                        tile_placed=Tile.get_starting_tile()._to_model()
                    )
                )

            self.state.start_new_round()

            for player_id in turn_order:
                if self.state.game_over:
                    break

                player = self.state.players[player_id]

                # If we are drawing the end of the river/base phase
                if not self.state.map.available_tiles:
                    self.state.tiles_exhausted = True

                    if self.state.river_phase:
                        self.complete_river_phase()

                    # Players have run out of tiles not in river phase
                    elif not player.tiles:
                        break

                    # No more draws but players can place tiles
                    else:
                        self.start_player_turn(player)
                        continue

                tiles_drawn = sample(
                    list(self.state.map.available_tiles), NUM_TILES_DRAWN_PER_ROUND
                )

                for tile in tiles_drawn:
                    self.state.map.available_tiles.remove(tile)
                    self.state.map.available_tiles_by_type[tile.tile_type].remove(tile)

                player.tiles.extend(tiles_drawn)
                self.mutator.commit(
                    EventPlayerDrewTiles(
                        player_id=player_id,
                        num_tiles=NUM_TILES_DRAWN_PER_ROUND,
                        tiles=[tile._to_model() for tile in tiles_drawn],
                    )
                )

                self.start_player_turn(player)

            # If mutator ended game
            if self.state.game_over:
                self.calc_final_points()

            if self.state.round > MAX_ROUNDS:
                self.mutator.commit(
                    EventGameEndedStaleMate(
                        reason="Reached maximum feasible round limit"
                    )
                )
                self.state.finalise_game()
                self.calc_final_points()

            if (
                self.state.tiles_exhausted
                and not self.state.river_phase
                and not any(p.tiles for p in self.state.players.values())
            ):
                self.mutator.commit(
                    EventGameEndedStaleMate(reason="All player tiles exhuasted")
                )
                self.state.finalise_game()
                self.calc_final_points()

    def start_player_turn(self, player: PlayerState) -> None:
        response = player.connection.query_place_tile(
            self.state, self.validator, self.censor
        )
        self.mutator.commit(response)

        # Tile placed ended the game
        if self.state.game_over:
            return

        response2 = player.connection.query_place_meeple(
            self.state, self.validator, self.censor
        )
        self.mutator.commit(response2)

    def complete_river_phase(self) -> None:
        self.state.start_base_phase()
        tile = self.state.map.placed_tiles[-1]

        edge: str
        for e, s in tile.internal_edges.items():
            assert tile.placed_pos is not None

            if s == StructureType.RIVER and not Tile.get_external_tile(
                e, tile.placed_pos, self.state.map._grid
            ):
                edge = e
                break

        else:
            assert False

        assert tile.placed_pos is not None
        x, y = tile.placed_pos

        river_end = Tile.get_river_end_tile()
        river_end.rotate_clockwise(TILE_EDGE_IDS[edge])
        x1, y1 = TILE_EXTERNAL_POS[edge](x, y)
        river_end.placed_pos = x1, y1

        self.state.map._grid[y1][x1] = river_end
        self.state.map.placed_tiles.append(river_end)

        print("River End Tile")
        self.mutator.commit(EventRiverPhaseCompleted(end_tile=river_end._to_model()))

        if EXPANSION:
            self.state.extend_base_phase()

        # Replinishes cards if moving to base phase or new game (river phase) this is before player draws tile for the round
        for player in self.state.players.values():
            tiles_drawn = sample(
                list(self.state.map.available_tiles), NUM_TILES_IN_HAND
            )
            self.state.map.available_tiles.difference_update(tiles_drawn)

            for tile in tiles_drawn:
                self.state.map.available_tiles_by_type[tile.tile_type].remove(tile)

            player.tiles.extend(tiles_drawn)

            self.mutator.commit(
                EventPlayerDrewTiles(
                    player_id=player.id,
                    num_tiles=NUM_TILES_IN_HAND,
                    tiles=[tile._to_model() for tile in tiles_drawn],
                )
            )

        self.state.tiles_exhausted = False
        self.state.river_phase = False

    def calc_final_points(self) -> None:
        tiles_unclaimed: list[tuple["Tile", str]] = [
            (meeple.placed, meeple.placed_edge)
            for player in self.state.players.values()
            for meeple in player.meeples
            if meeple.placed is not None
        ]

        # Players may have multiple meepls per connected structure component
        structures_visited: set[tuple["Tile", str]] = set()

        for tile, edge in tiles_unclaimed:
            if edge == MONASTARY_IDENTIFIER:
                assert tile.placed_pos

                meeple = tile.internal_claims[edge]
                subsribers = [
                    cast(MonastaryNeighbourSubsciber, s)
                    for s in self.state.tile_publisher.watchers[tile.placed_pos]
                    if isinstance(s, MonastaryNeighbourSubsciber)
                    and cast(MonastaryNeighbourSubsciber, s).center == tile.placed_pos
                ]

                assert len(subsribers) == 1

                assert meeple is not None

                reward = len(subsribers[0].filled)
                self.state.players[meeple.player_id].points += reward

                meeple._free_meeple()
                self.mutator.commit(
                    EventPlayerMeepleFreed(
                        player_id=meeple.player_id,
                        reward=reward,
                        tile=tile._to_model(),
                        placed_on=edge,
                    )
                )

                continue

            # Only check reward once
            if (tile, edge) in structures_visited:
                continue

            # Redundant safe add
            structures_visited.add((tile, edge))

            players = self.state._get_claims_objs(tile, edge)
            players_meeples = sorted(players.values(), key=len, reverse=True)
            for player_meeples in players_meeples:
                for m in player_meeples:
                    assert m.placed is not None
                    structures_visited.add((m.placed, m.placed_edge))

            partial_rewarded_meeples = [players_meeples[0][0]]
            returning_meeples = []

            assert (
                partial_rewarded_meeples[0].placed is not None
                and partial_rewarded_meeples[0].placed_edge != ""
            )

            for pm in players_meeples[1:]:
                if pm and len(pm) == len(players_meeples[0]):
                    partial_rewarded_meeples.append(pm[0])

                elif pm:
                    returning_meeples.append(pm[0])

                for m in pm:
                    assert m.placed is not None and m.placed_edge != ""

            reward = self.state._get_reward(tile, edge, partial=True)

            for meeple in partial_rewarded_meeples:
                self.state.players[meeple.player_id].points += reward
                assert meeple.placed
                self.mutator.commit(
                    EventPlayerMeepleFreed(
                        player_id=meeple.player_id,
                        reward=reward,
                        tile=meeple.placed._to_model(),
                        placed_on=edge,
                    )
                )
                meeple._free_meeple()

            for meeple in returning_meeples:
                assert meeple.placed
                self.mutator.commit(
                    EventPlayerMeepleFreed(
                        player_id=meeple.player_id,
                        reward=NO_POINTS,
                        tile=meeple.placed._to_model(),
                        placed_on=edge,
                    )
                )
                meeple._free_meeple()

        player, points = self.state.get_player_points()[0]
        self.mutator.commit(EventPlayerWon(player_id=player, points=points))

    def finish(self) -> None:
        # Write the result.
        inspector = EventInspector(
            self.state.event_history,
            {i: j for i, j in self.state.get_player_points()},
            self.state.get_rankings(),
        )
        result = inspector.get_result()

        with open(f"{CORE_DIRECTORY}/output/results.json", "w") as f:
            f.write(result.model_dump_json())

        # Write the game log.
        with open(f"{CORE_DIRECTORY}/output/game.json", "w") as f:
            f.write(inspector.get_recording_json())

        visualiser_data = inspector.get_visualiser_json()
        with open(
            f"{CORE_DIRECTORY}/output/visualiser_forwards_differential.json", "w"
        ) as f:
            f.write(visualiser_data)

        def copy_stdout_stderr_player(player: int) -> None:
            stderr_path = f"{CORE_DIRECTORY}/submission{player}/io/submission.err"
            stderr_path_new = f"{CORE_DIRECTORY}/output/submission_{player}.err"
            stdout_path = f"{CORE_DIRECTORY}/submission{player}/io/submission.log"
            stdout_path_new = f"{CORE_DIRECTORY}/output/submission_{player}.log"

            try:
                shutil.copy(stderr_path, stderr_path_new, follow_symlinks=False)
            except (FileNotFoundError, IsADirectoryError, FileExistsError):
                with open(stderr_path_new, "w") as f:
                    f.write(
                        "Your submission.err file is either missing or is a directory."
                    )

            try:
                shutil.copy(stdout_path, stdout_path_new, follow_symlinks=False)
            except (FileNotFoundError, IsADirectoryError, FileExistsError):
                with open(stdout_path_new, "w") as f:
                    f.write(
                        "Your submission.log file is either missing or is a directory."
                    )

        # Only copy for the player who was banned, otherwise copy for all players, or only copy the log
        # if the match was cancelled.
        print(f"[engine]: match complete, outcome was {{{result}}}", flush=True)
        match result:
            case GameBanResult() as x:
                copy_stdout_stderr_player(player=x.player)

            case GameSuccessResult():
                for player in self.state.players.keys():
                    copy_stdout_stderr_player(player)

            case GameCancelledResult():
                pass
