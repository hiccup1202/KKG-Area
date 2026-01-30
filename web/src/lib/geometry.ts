export type Point = { x: number; y: number };
export type Segment = { a: Point; b: Point };
export type Interval = { start: number; end: number };

function sub(a: Point, b: Point): Point {
  return { x: a.x - b.x, y: a.y - b.y };
}

function dot(a: Point, b: Point): number {
  return a.x * b.x + a.y * b.y;
}

function cross(a: Point, b: Point): number {
  return a.x * b.y - a.y * b.x;
}

function norm(a: Point): number {
  return Math.hypot(a.x, a.y);
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

function pointLinePerpDistance(p: Point, a: Point, b: Point): number {
  // Distance from p to infinite line through a-b
  const ab = sub(b, a);
  const ap = sub(p, a);
  const denom = norm(ab);
  if (denom < 1e-9) return norm(ap);
  return Math.abs(cross(ap, ab)) / denom;
}

export function segmentsFromPolyline(points: Point[], closed: boolean): Segment[] {
  const segs: Segment[] = [];
  for (let i = 0; i + 1 < points.length; i++) {
    segs.push({ a: points[i], b: points[i + 1] });
  }
  if (closed && points.length >= 2) {
    segs.push({ a: points[points.length - 1], b: points[0] });
  }
  return segs;
}

export function mergeIntervals(intervals: Interval[]): Interval[] {
  const cleaned = intervals
    .map((i) => ({ start: Math.min(i.start, i.end), end: Math.max(i.start, i.end) }))
    .filter((i) => i.end > i.start);
  cleaned.sort((a, b) => a.start - b.start);
  const out: Interval[] = [];
  for (const cur of cleaned) {
    const last = out[out.length - 1];
    if (!last || cur.start > last.end) out.push({ ...cur });
    else last.end = Math.max(last.end, cur.end);
  }
  return out;
}

export function sumIntervals(intervals: Interval[]): number {
  return intervals.reduce((acc, i) => acc + (i.end - i.start), 0);
}

/**
 * Returns overlap interval of segB projected onto segA (t along segA: 0..lenA),
 * when they are approximately colinear and close; otherwise null.
 */
export function overlapIntervalOnSegment(
  segA: Segment,
  segB: Segment,
  opts: { distanceTolPx: number; angleTolDeg: number; minOverlapPx: number },
): Interval | null {
  const A = sub(segA.b, segA.a);
  const B = sub(segB.b, segB.a);
  const lenA = norm(A);
  const lenB = norm(B);
  if (lenA < 1e-6 || lenB < 1e-6) return null;

  // Angle check (direction-agnostic)
  const cos = Math.abs(dot(A, B) / (lenA * lenB));
  const angle = (Math.acos(clamp(cos, -1, 1)) * 180) / Math.PI;
  if (angle > opts.angleTolDeg) return null;

  // Distance to segA's infinite line
  const d1 = pointLinePerpDistance(segB.a, segA.a, segA.b);
  const d2 = pointLinePerpDistance(segB.b, segA.a, segA.b);
  if (Math.max(d1, d2) > opts.distanceTolPx) return null;

  // Project B endpoints onto A axis
  const u = { x: A.x / lenA, y: A.y / lenA };
  const t1 = dot(sub(segB.a, segA.a), u);
  const t2 = dot(sub(segB.b, segA.a), u);

  const start = Math.max(0, Math.min(t1, t2));
  const end = Math.min(lenA, Math.max(t1, t2));
  if (end - start < opts.minOverlapPx) return null;
  return { start, end };
}

