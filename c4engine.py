from typing import List


class C4Engine:
    WIDTH = 7
    HEIGHT = 6
    START = 'S'
    EMPTY = '-'
    PLAYERS = 'AB'
    DRAW = 'D'
    RESULT_TYPES = PLAYERS + DRAW
    MOVES = list(START + RESULT_TYPES) + [str(i) for i in range(WIDTH)]

    def __init__(self, start_seq: str = ''):
        self.reset()
        for move in start_seq:
            self.make_move(move)


    def reset(self) -> None:
        self._board = [[C4Engine.EMPTY for x in range(C4Engine.WIDTH)] for y in range(C4Engine.HEIGHT)]
        self._game_started = False
        self._result = None
        self._turn = 0


    def board(self) -> List[List[str]]:
        return self._board


    def result(self) -> str | None:
        return self._result


    def player_to_move(self) -> str:
        return C4Engine.PLAYERS[self._turn]


    def is_legal_move(self, move: str) -> bool:
        if move not in C4Engine.MOVES:
            return False

        if not self._game_started:
            return move == C4Engine.START

        if move == C4Engine.START:
            return False

        if move in C4Engine.RESULT_TYPES:
            return move == self._result

        if self._result is not None:
            return False

        return self._board[0][int(move)] == C4Engine.EMPTY


    def make_move(self, move: str) -> bool:
        if not self.is_legal_move(move):
            return False

        if move == C4Engine.START:
            self._game_started = True
            return True

        if move in C4Engine.RESULT_TYPES:
            return True

        x = int(move)
        for y, row in enumerate(self._board[::-1], 1):
            if row[x] == C4Engine.EMPTY:
                row[x] = C4Engine.PLAYERS[self._turn]
                break
        self._last_y = C4Engine.HEIGHT - y
        self._result = self._get_result(x, self._last_y)
        self._turn = 1 - self._turn

        return True


    def _get_result(self, move_x: int, move_y: int) -> str | None:
        player = self._board[move_y][move_x]
        dirs = [(0,1), (1,0), (1,1), (1,-1)]

        def in_bounds(x, y):
            return 0 <= x < C4Engine.WIDTH and 0 <= y < C4Engine.HEIGHT

        for dx, dy in dirs:
            start_d = max(d for d in range(4) if in_bounds(move_x-d*dx, move_y-d*dy))
            end_d = max(d for d in range(4) if in_bounds(move_x+d*dx, move_y+d*dy))
            in_row = 0
            for d in range(-start_d, end_d+1):
                x = move_x + d*dx
                y = move_y + d*dy
                if self._board[y][x] == player:
                    in_row += 1
                else:
                    in_row = 0

                if in_row == 4:
                    return player

        return C4Engine.DRAW if C4Engine.EMPTY not in self._board[0] else None
