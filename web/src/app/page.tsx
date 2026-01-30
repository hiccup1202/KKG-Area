"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./page.module.css";
import type { KkgContour, KkgJson } from "@/lib/kkgTypes";
import { parseKkgJson } from "@/lib/kkgTypes";
import {
  mergeIntervals,
  overlapIntervalOnSegment,
  segmentsFromPolyline,
  sumIntervals,
  type Segment,
} from "@/lib/geometry";

type RoomResult = {
  roomId: number;
  roomName: string;
  wallLengthPx: number;
  doorLengthPx: number;
  wallAreaM2: number;
};

const SCALE_PX = 96.7;
const SCALE_M = 1.365;
const CEILING_H_M = 2.4;
const DOOR_H_M = 2.0;

function fmt(n: number, digits = 2) {
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(digits);
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "").trim();
  if (h.length !== 6) return `rgba(0,0,0,${alpha})`;
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function toSegments(contours: KkgContour[], labelId: number): Segment[] {
  const segs: Segment[] = [];
  for (const c of contours) {
    if (c.label_id !== labelId) continue;
    const pts = c.vertices;
    // walls/doors are expected to be lines, but we accept polylines.
    for (const s of segmentsFromPolyline(pts, false)) segs.push(s);
  }
  return segs;
}

export default function Home() {
  const [data, setData] = useState<KkgJson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedRoomId, setSelectedRoomId] = useState<number | null>(null);
  const [imageLoadNonce, setImageLoadNonce] = useState(0);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const imageUrl = useMemo(() => {
    if (!data?.image_data?.base64 || !data?.image_data?.mime_type) return null;
    return `data:${data.image_data.mime_type};base64,${data.image_data.base64}`;
  }, [data]);

  const rooms = useMemo(() => {
    if (!data) return [];
    // Prefer room_name!=null; fallback to label_id==1.
    const byName = data.contours.filter((c) => c.room_name);
    const src = byName.length > 0 ? byName : data.contours.filter((c) => c.label_id === 1);
    return src;
  }, [data]);

  const wallSegments = useMemo(() => (data ? toSegments(data.contours, 2) : []), [data]);
  const doorSegments = useMemo(() => (data ? toSegments(data.contours, 3) : []), [data]);

  const results: RoomResult[] = useMemo(() => {
    if (!data) return [];

    const mPerPx = SCALE_M / SCALE_PX;
    const distanceTolPx = 6; // tolerance for alignment (px)
    const angleTolDeg = 8;
    const minOverlapPx = 6;

    const computeForRoom = (room: KkgContour): RoomResult => {
      const ring = room.vertices;
      const roomEdges = segmentsFromPolyline(ring, true);

      let wallLenPx = 0;
      let doorLenPx = 0;

      for (const edge of roomEdges) {
        const wallIntervals = [];
        const doorIntervals = [];

        for (const ws of wallSegments) {
          const interval = overlapIntervalOnSegment(edge, ws, {
            distanceTolPx,
            angleTolDeg,
            minOverlapPx,
          });
          if (interval) wallIntervals.push(interval);
        }

        for (const ds of doorSegments) {
          const interval = overlapIntervalOnSegment(edge, ds, {
            distanceTolPx,
            angleTolDeg,
            minOverlapPx,
          });
          if (interval) doorIntervals.push(interval);
        }

        wallLenPx += sumIntervals(mergeIntervals(wallIntervals));
        doorLenPx += sumIntervals(mergeIntervals(doorIntervals));
      }

      const wallLenM = wallLenPx * mPerPx;
      const doorLenM = doorLenPx * mPerPx;
      const wallAreaM2 = Math.max(0, wallLenM * CEILING_H_M - doorLenM * DOOR_H_M);

      return {
        roomId: room.id,
        roomName: room.room_name ?? `Room ${room.id}`,
        wallLengthPx: wallLenPx,
        doorLengthPx: doorLenPx,
        wallAreaM2,
      };
    };

    return rooms.map(computeForRoom).sort((a, b) => b.wallAreaM2 - a.wallAreaM2);
  }, [data, rooms, wallSegments, doorSegments]);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !data) return;
    if (img.naturalWidth === 0 || img.naturalHeight === 0) return;

    const rect = img.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const sx = rect.width / img.naturalWidth;
    const sy = rect.height / img.naturalHeight;

    const drawContour = (c: KkgContour) => {
      const pts = c.vertices;
      if (pts.length < 2) return;
      const isRoom = c.room_name != null || c.label_id === 1;
      const isWall = c.label_id === 2;
      const isDoor = c.label_id === 3;

      const stroke =
        (isDoor && "rgba(255, 140, 0, 0.95)") ||
        (isWall && "rgba(0, 120, 255, 0.9)") ||
        (c.color ? hexToRgba(c.color, 0.9) : "rgba(0,0,0,0.7)");
      const fill =
        isRoom && (c.color ? hexToRgba(c.color, 0.12) : "rgba(60,60,60,0.08)");

      const lineW = isDoor ? 3 : isWall ? 2 : 1;

      ctx.beginPath();
      ctx.moveTo(pts[0].x * sx, pts[0].y * sy);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * sx, pts[i].y * sy);
      if (isRoom) ctx.closePath();

      if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
      }
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineW;
      ctx.stroke();
    };

    // Draw non-selected first, then selected room highlight
    for (const c of data.contours) drawContour(c);

    if (selectedRoomId != null) {
      const room = rooms.find((r) => r.id === selectedRoomId);
      if (room) {
        const pts = room.vertices;
        if (pts.length >= 3) {
          ctx.beginPath();
          ctx.moveTo(pts[0].x * sx, pts[0].y * sy);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * sx, pts[i].y * sy);
          ctx.closePath();
          ctx.strokeStyle = "rgba(255,0,0,0.95)";
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      }
    }
  }, [data, rooms, selectedRoomId, imageUrl, imageLoadNonce]);

  const onFile = async (file: File | null) => {
    if (!file) return;
    setError(null);
    const text = await file.text();
    try {
      const parsed = parseKkgJson(text);
      setData(parsed);
      setError(null);
      setSelectedRoomId(null);
      setImageLoadNonce((n) => n + 1);
    } catch (e) {
      setData(null);
      setSelectedRoomId(null);
      setError(e instanceof Error ? e.message : String(e));
      setImageLoadNonce((n) => n + 1);
    }
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <header className={styles.header}>
          <div>
            <h1 className={styles.title}>KKG Wall Area Calculator</h1>
            <p className={styles.subtitle}>
              Upload `部屋・壁・ドアマスク.json` and get wall area per room (door deducted).
            </p>
          </div>
          <div className={styles.controls}>
            <label className={styles.fileButton}>
              <input
                type="file"
                accept="application/json,.json"
                onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
              />
              Choose JSON
            </label>
            <button
              className={styles.secondaryButton}
              onClick={() => {
                setData(null);
                setError(null);
                setSelectedRoomId(null);
                imgRef.current = null;
                setImageLoadNonce((n) => n + 1);
              }}
              type="button"
            >
              Clear
            </button>
          </div>
        </header>

        {error && <div className={styles.error}>Error: {error}</div>}

        <section className={styles.grid}>
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>Preview</h2>
              <div className={styles.meta}>
                <span>
                  Scale: {SCALE_PX}px = {SCALE_M}m
                </span>
                <span>Ceiling: {CEILING_H_M}m</span>
                <span>Door height: {DOOR_H_M}m</span>
              </div>
            </div>

            {imageUrl ? (
              <div className={styles.previewWrap}>
                <Image
                  src={imageUrl}
                  alt="floorplan"
                  className={styles.previewImg}
                  width={data?.image_width ?? 1200}
                  height={data?.image_height ?? 800}
                  style={{ width: "100%", height: "auto" }}
                  unoptimized
                  onLoadingComplete={(img) => {
                    imgRef.current = img;
                    setImageLoadNonce((n) => n + 1);
                  }}
                />
                <canvas ref={canvasRef} className={styles.overlayCanvas} />
              </div>
            ) : (
              <div className={styles.placeholder}>
                Upload a JSON that contains `image_data.base64` to see the preview.
              </div>
            )}
          </div>

          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>Wall area per room</h2>
              <div className={styles.meta}>
                <span>Rooms: {results.length}</span>
                <span>Walls segments: {wallSegments.length}</span>
                <span>Door segments: {doorSegments.length}</span>
              </div>
            </div>

            {results.length === 0 ? (
              <div className={styles.placeholder}>Upload a JSON to compute results.</div>
            ) : (
              <div className={styles.tableWrap}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Room</th>
                      <th className={styles.num}>Wall area (m²)</th>
                      <th className={styles.num}>Wall length (m)</th>
                      <th className={styles.num}>Door length (m)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r) => {
                      const mPerPx = SCALE_M / SCALE_PX;
                      const wallM = r.wallLengthPx * mPerPx;
                      const doorM = r.doorLengthPx * mPerPx;
                      const selected = selectedRoomId === r.roomId;
                      return (
                        <tr
                          key={r.roomId}
                          className={selected ? styles.selectedRow : undefined}
                          onClick={() => setSelectedRoomId(r.roomId)}
                          role="button"
                          tabIndex={0}
                        >
                          <td>{r.roomName}</td>
                          <td className={styles.num}>{fmt(r.wallAreaM2, 2)}</td>
                          <td className={styles.num}>{fmt(wallM, 2)}</td>
                          <td className={styles.num}>{fmt(doorM, 2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className={styles.hint}>
                  Tip: click a row to highlight that room’s polygon in red.
                </p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
