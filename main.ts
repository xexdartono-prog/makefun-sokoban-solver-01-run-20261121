// Sokoban solver: push-optimal, room by room. Node 22+, stdlib only.
// A level is one hallway with rooms off it; a box pushed out of a room can never
// come back, so every box stays in its room and the level's optimum is the sum of
// its rooms' optima. Each room is solved on its own by a breadth-first search over
// pushes (fewest pushes first, dead-square and frozen-block pruning), then the
// rooms' moves are stitched with walks through the hallway. Deterministic: fixed
// work budgets, fixed orders, no clock, no randomness.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const ROOM_STATES = 250000; // states one room search keeps, a work budget
let budget = 6000000; // states the whole run keeps, a work budget

type Bd = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };
type Room = { boxes: number[]; goals: number[]; exits: Set<number> };
const DIRS = [[0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r']] as const;

function parse(text: string): Bd {
  const rows = text.split('\n').filter((r) => r.length);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares a box can be pushed from onto some goal: pull back from the goals.
// A square y is live when a box on y pushes to a live square with the player
// standing on the square behind it.
function live(b: Bd): Uint8Array {
  const lv = new Uint8Array(b.w * b.h);
  const q = [...b.goals];
  for (const g of b.goals) lv[g] = 1;
  for (let i = 0; i < q.length; i++) {
    const x = q[i];
    for (const [dx, dy] of DIRS) {
      const d = dy * b.w + dx;
      const y = x - d;
      if (!b.wall[y] && !b.wall[y - d] && !lv[y]) { lv[y] = 1; q.push(y); }
    }
  }
  return lv;
}

// The squares the player reaches from `from` with `occ` in the way; returns the lowest.
function region(b: Bd, occ: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const w = b.w;
    let q = p - w;
    if (!b.wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p + w;
    if (!b.wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p - 1;
    if (!b.wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p + 1;
    if (!b.wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
  }
  return low;
}

// Whether the player reaches any exit square from `from` with `occ` in the way.
function reachesExit(b: Bd, occ: Uint8Array, from: number, exits: Set<number>, seen: Int32Array, stamp: number, stack: Int32Array): boolean {
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (exits.has(p)) return true;
    const w = b.w;
    let q = p - w;
    if (!b.wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p + w;
    if (!b.wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p - 1;
    if (!b.wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p + 1;
    if (!b.wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
  }
  return false;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and
// boxes with a box off its goal.
function frozen(b: Bd, occ: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - b.w, t - b.w - 1]) {
    const e = a + 1;
    const c = a + b.w;
    const d = c + 1;
    if ((b.wall[a] || occ[a]) && (b.wall[e] || occ[e]) && (b.wall[c] || occ[c]) && (b.wall[d] || occ[d])) {
      if ((occ[a] && !goal[a]) || (occ[e] && !goal[e]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
    }
  }
  return false;
}

// The shortest walk from `from` to `to` around walls and boxes, lower LURD
// letters, or null when `to` is out of reach.
function walkTo(b: Bd, occ: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(b.wall.length).fill(-1);
  const how: string[] = new Array(b.wall.length);
  const q = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const p = q[i];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const n = p + dy * b.w + dx;
      if (b.wall[n] || occ[n] || prev[n] !== -1) continue;
      prev[n] = p;
      how[n] = letter;
      q.push(n);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// The push-optimal solution of one board: {pushes, moves, end} or null when the
// search gives up. A state is the boxes' squares and the player's reachable
// region (named by its lowest square); states expand in order of pushes made,
// so the first solved state met is push-optimal.
function solveRoom(sub: Bd, exits: Set<number> | null, needExit: boolean): { pushes: number; moves: string; end: number } | null {
  const { w, wall, goals } = sub;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const lv = live(sub);
  if (sub.boxes.some((b) => !lv[b])) return null;
  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  const exitOk = (bs: number[], p: number): boolean => {
    if (!needExit || exits === null) return true;
    for (const b of bs) occ[b] = 1;
    const ok = reachesExit(sub, occ, p, exits, mark, ++stamp, stack);
    for (const b of bs) occ[b] = 0;
    return ok;
  };
  const sBoxes: number[][] = [sub.boxes.slice()];
  const sPlayer: number[] = [sub.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  // A state's key: the region's lowest square and the boxes' squares. Numeric
  // while the digits fit in a double exactly, a string otherwise.
  const numeric = (sub.boxes.length + 1) * Math.log2(size) < 52;
  const keyOf = (bs: number[], low: number): number | string => {
    if (!numeric) return low + '|' + bs.join(',');
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };
  const keys = new Map<number | string, number>();
  for (const b of sub.boxes) occ[b] = 1;
  const low0 = region(sub, occ, sub.player, seen, ++stamp, stack);
  for (const b of sub.boxes) occ[b] = 0;
  keys.set(keyOf(sub.boxes, low0), 0);
  let found = solved(sub.boxes) && exitOk(sub.boxes, sub.player) ? 0 : -1;
  for (let head = 0; found < 0 && head < sBoxes.length; head++) {
    const boxes = sBoxes[head];
    for (const b of boxes) occ[b] = 1;
    region(sub, occ, sPlayer[head], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !lv[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        const fr = frozen(sub, occ, goal, t);
        let key: number | string = '';
        let next: number[] | null = null;
        if (!fr) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, region(sub, occ, b, mark, ++stamp, stack));
        }
        occ[t] = 0;
        occ[b] = 1;
        if (fr || keys.has(key)) continue;
        if (sBoxes.length >= ROOM_STATES || --budget <= 0) {
          for (const x of boxes) occ[x] = 0;
          return null;
        }
        const idx = sBoxes.length;
        keys.set(key, idx);
        sBoxes.push(next!);
        sPlayer.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushTo.push(t);
        if (solved(next!) && exitOk(next!, b)) { found = idx; break; }
      }
    }
    for (const x of boxes) occ[x] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = sub.player;
  let boxes = sub.boxes.slice();
  for (const s of path) {
    const next = sBoxes[s];
    const from = boxes.find((b) => !next.includes(b))!;
    const to = next.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occ[b] = 1;
    const walk = walkTo(sub, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    if (walk === null) return null;
    moves += walk + dir[2];
    player = from;
    boxes = next;
  }
  return { pushes: path.length, moves, end: player };
}

// The level's rooms: cut the dead squares of the player's row (the hallway: a
// box there reaches no goal) out of the floor; each part left that holds a box
// is a room, with its exit squares (next to the hallway). Null when the level
// is no set of rooms.
function splitRooms(bd: Bd): Room[] | null {
  const lv = live(bd);
  const row = Math.floor(bd.player / bd.w);
  const cut = new Uint8Array(bd.wall.length);
  for (let i = row * bd.w; i < (row + 1) * bd.w; i++) if (!bd.wall[i] && !lv[i]) cut[i] = 1;
  if (bd.boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(bd.wall.length).fill(-1);
  const rooms: { cells: number[]; boxes: number[]; goals: number[]; exits: Set<number> }[] = [];
  for (let s = 0; s < bd.wall.length; s++) {
    if (bd.wall[s] || cut[s] || part[s] >= 0) continue;
    const cells = [s];
    part[s] = rooms.length;
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      for (const [dx, dy] of DIRS) {
        const q = c + dy * bd.w + dx;
        if (!bd.wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; cells.push(q); }
      }
    }
    rooms.push({ cells, boxes: [], goals: [], exits: new Set() });
  }
  if (part[bd.player] >= 0) return null;
  for (const b of bd.boxes) rooms[part[b]].boxes.push(b);
  for (const g of bd.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  for (const r of rooms) {
    for (const c of r.cells) {
      for (const [dx, dy] of DIRS) {
        if (cut[c + dy * bd.w + dx]) { r.exits.add(c); break; }
      }
    }
  }
  return rooms.filter((r) => r.boxes.length > 0);
}

// One room-by-room attempt: each room's push-optimal solution in turn, the
// other rooms' boxes standing as walls (on their goals once their room is
// done), the player walking on from where the last room left it.
function stitch(bd: Bd, rooms: Room[], reverse: boolean, needExit: boolean): string | null {
  const order = reverse ? [...rooms].reverse() : rooms;
  const done: Room[] = [];
  let player = bd.player;
  let moves = '';
  for (let k = 0; k < order.length; k++) {
    const room = order[k];
    const wall = bd.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of done.includes(other) ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Bd = {
      w: bd.w, h: bd.h, wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const res = solveRoom(sub, room.exits, needExit && k < order.length - 1);
    if (res === null) return null;
    moves += res.moves;
    player = res.end;
    done.push(room);
  }
  return moves;
}

// Replay the moves on the level: true when every step is legal and every box
// ends on a goal.
function replayOk(bd: Bd, moves: string): boolean {
  const box = new Uint8Array(bd.wall.length);
  for (const b of bd.boxes) box[b] = 1;
  const goal = new Uint8Array(bd.wall.length);
  for (const g of bd.goals) goal[g] = 1;
  let p = bd.player;
  for (const ch0 of moves) {
    const ch = ch0.toLowerCase();
    const d = ch === 'u' ? -bd.w : ch === 'd' ? bd.w : ch === 'l' ? -1 : 1;
    const q = p + d;
    if (bd.wall[q]) return false;
    if (box[q]) {
      const t = q + d;
      if (bd.wall[t] || box[t]) return false;
      box[q] = 0;
      box[t] = 1;
    }
    p = q;
  }
  for (let i = 0; i < box.length; i++) if (box[i] && !goal[i]) return false;
  return true;
}

function solveLevel(text: string): string {
  const bd = parse(text);
  const rooms = splitRooms(bd);
  if (rooms !== null) {
    // Each attempt is tried in turn and kept only when its replay solves the
    // level, so the usual case stops after the first one.
    const a = stitch(bd, rooms, false, true);
    if (a !== null && replayOk(bd, a)) return a;
    const b2 = stitch(bd, rooms, false, false);
    if (b2 !== null && replayOk(bd, b2)) return b2;
    const c = stitch(bd, rooms, true, true);
    if (c !== null && replayOk(bd, c)) return c;
  }
  const whole = solveRoom(bd, null, false);
  if (whole !== null && replayOk(bd, whole.moves)) return whole.moves;
  return '';
}

for (const id of readdirSync(IN).sort()) {
  const moves = solveLevel(readFileSync(join(IN, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(OUT, id, 'solution.txt'), moves + '\n');
}
