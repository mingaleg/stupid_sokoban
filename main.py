from enum import Enum
from copy import deepcopy
from typing import Dict, Generic, TypeVar, Optional, Iterable
from abc import ABCMeta, abstractmethod
from functools import cached_property
from collections import deque

T = TypeVar("T")
U = TypeVar("U")

class Direction(Enum):
    UP    = (-1, 0)
    DOWN  = (+1, 0)
    LEFT  = (0, -1)
    RIGHT = (0, +1)

class Index(tuple[int, int]):
    def add_direction(self, d: Direction) -> 'Index':
        x, y = self
        dx, dy = d.value
        return Index((x+dx, y+dy))

class Grid(Generic[T], metaclass=ABCMeta):
    n: int
    m: int
    _grid: list[list[T]]

    @property
    @abstractmethod
    def out_of_bounds_val(self) -> T:
        pass

    def __init__(self, grid: list[list[T]]):
        assert (n := len(grid)) > 0
        m = len(grid)
        for row in grid:
            assert len(row) == m
        self._grid = grid
        self.n, self.m = n, m

    def in_bounds(self, ind) -> bool:
        x, y = ind
        return (0 <= x < self.n) and (0 <= y < self.m)
    
    def __getitem__(self, ind: Index) -> T:
        if not self.in_bounds(ind):
            return self.out_of_bounds_val
        x, y = ind
        return self._grid[x][y]

    def __setitem__(self, ind: Index, val: T):
        x, y = ind
        self._grid[x][y] = val

    def items(self) -> Iterable[tuple[Index, T]]:
        for x, row in enumerate(self._grid):
            for y, el in enumerate(row):
                yield Index((x, y)), el

    def values(self) -> Iterable[T]:
        for row in self._grid:
            yield from row

    def __eq__(self, other):
        return all(x==y for x,y in zip(self.values(), other.values()))

class Cell(Enum):
    EMPTY = 'e'
    TARGET = 't'
    WALL = 'w'

class CellsGrid(Grid[Cell]):
    out_of_bounds_val = Cell.WALL

class Sprite(Enum):
    NULL = 'n'
    PLAYER = 'p'
    BOX = 'b'

class SpritesGrid(Grid[Sprite]):
    out_of_bounds_val = Sprite.BOX
    _player: Index

    def __init__(self, grid: list[list[Sprite]]):
        super().__init__(grid)
        _player: Optional[Index] = None
        for x, row in enumerate(grid):
            for y, sprite in enumerate(row):
                if sprite != Sprite.PLAYER:
                    continue
                assert _player is None, "several players"
                _player = Index((x, y))
        assert _player is not None, "no players"
        self._player = _player

    def _swap(self, a: Index, b:Index) -> Optional['SpritesGrid']:
        if not self.in_bounds(a) or not self.in_bounds(b):
            return None
        _player = self._player
        if _player == a:
            _player = b
        elif _player == b:
            _player = a
        new = deepcopy(self)
        new[a], new[b] = new[b], new[a]
        new._player = _player
        return new

    def move(self, d: Direction) -> Optional['SpritesGrid']:
        next1 = self._player.add_direction(d)
        next2 = next1.add_direction(d)
        if self[next1] == Sprite.NULL:
            return self._swap(self._player, next1)
        if self[next1] == Sprite.BOX and self[next2] == Sprite.NULL:
            new = self._swap(next1, next2)
            if new is None:
                return None
            return new._swap(self._player, next1)
        return None

    def __hash__(self):
        return hash((self._player, tuple(idx for idx, el in self.items() if el == Sprite.BOX)))

class Bijection(Generic[T, U]):
    _forwards: dict[T, U]
    _backwards: dict[U, T]

    def __init__(self, *items: tuple[T, U]):
        cnt = 0
        self._forwards = {}
        self._backwards = {}
        for k, v in items:
            cnt += 1
            self._forwards[k] = v
            self._backwards[v] = k
        assert len(self._forwards) == cnt
        assert len(self._backwards) == cnt

    @classmethod
    def _with(cls, forwards: dict[T, U], backwards: dict[U, T]):
        self = cls.__new__(cls)
        self._forwards = forwards
        self._backwards = backwards
        return self

    @property
    def rev(self) -> 'Bijection[U, T]':
        return Bijection._with(self._backwards, self._forwards)

    def __getitem__(self, idx: T) -> U:
        return self._forwards[idx]

class State:
    _cells: CellsGrid
    _sprites: SpritesGrid
    _parent: Optional['State']

    char_to_enums = Bijection(
        ('.', (Cell.EMPTY,  Sprite.NULL)),
        ('#', (Cell.WALL,   Sprite.NULL)),
        ('X', (Cell.TARGET, Sprite.PLAYER)),
        ('@', (Cell.TARGET, Sprite.BOX)),
        ('*', (Cell.EMPTY,  Sprite.PLAYER)),
        ('0', (Cell.EMPTY,  Sprite.BOX)),
        ('_', (Cell.TARGET, Sprite.NULL)),
    )

    def __init__(self, cells: CellsGrid, sprites: SpritesGrid, parent: Optional['State'] = None):
        self._cells = cells
        self._sprites = sprites
        self._parent = parent

    @cached_property
    def final(self) -> bool:
        for c, s in zip(self._cells.values(), self._sprites.values()):
            if c == Cell.TARGET and s != Sprite.BOX:
                    return False
            if c == Cell.WALL and s != Sprite.NULL:
                    return False
        return True

    @cached_property
    def valid(self) -> bool:
        for c, s in zip(self._cells.values(), self._sprites.values()):
            if c == Cell.WALL and s != Sprite.NULL:
                    return False
        return True

    def next_states(self) -> Iterable['State']:
        for d in Direction:
            new_sprites = self._sprites.move(d)
            if new_sprites is None:
                continue
            new_state = State(self._cells, new_sprites, self)
            if new_state.valid:
                yield new_state

    def __hash__(self):
        return hash(self._sprites)

    def path(self) -> Iterable['State']:
        if self._parent is not None:
            yield from self._parent.path()
        yield self

    def __eq__(self, other):
        return self._sprites == other._sprites

    def __repr__(self):
        out = [['' for y in range(self._cells.m)] for x in range(self._cells.n)]
        for ((cx, cy), c), ((sx, sy), s) in zip(self._cells.items(), self._sprites.items()):
            out[cx][cy] = self.char_to_enums.rev[(c, s)]
        field = '\n  '.join(''.join(row) for row in out)
        return 'State<\n  ' + field + '\n>'

    @classmethod
    def from_string(cls, s):
        cells = []
        sprites = []
        empty_line = True
        for ch in s + '\n':
            if ch == ' ':
                continue
            if ch == '\n':
                empty_line = True
                continue
            if empty_line:
                cells.append([])
                sprites.append([])
            empty_line = False
            nc, ns = cls.char_to_enums[ch]
            cells[-1].append(nc)
            sprites[-1].append(ns)
        return cls(
            CellsGrid(cells),
            SpritesGrid(sprites),
        )

    def shortest_final(self) -> Optional['State']:
        assert self.valid
        queue: deque[State] = deque((self,))
        visited: set[int] = set((hash(self),))
        while queue:
            state = queue.popleft()
            if state.final:
                return state
            for new_state in state.next_states():
                h = hash(new_state)
                if h not in visited:
                    queue.append(new_state)
                    visited.add(h)
        return None

if __name__ == '__main__':
    final = State.from_string('''
        #...#
        .0.0_
        *.#._
        .0.0_
        #..._
    ''').shortest_final()
    assert final is not None
    for s in final.path():
        print(s)

