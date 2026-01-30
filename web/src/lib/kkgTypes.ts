export type KkgVertex = { x: number; y: number };

export type KkgContour = {
  id: number;
  vertices: KkgVertex[];
  color?: string | null;
  room_name?: string | null;
  area?: number;
  perimeter?: number;
  label_id: number; // 1: room, 2: wall, 3: door (per assignment)
  holes?: unknown;
  isLine?: boolean | null;
  lineWidth?: number | null;
  instanceGroupId?: unknown;
};

export type KkgJson = {
  contours: KkgContour[];
  image_data?: { base64: string; mime_type: string } | null;
  image_width?: number;
  image_height?: number;
  timestamp?: string;
};

export function parseKkgJson(text: string): KkgJson {
  const raw = JSON.parse(text) as unknown;
  if (!raw || typeof raw !== "object") throw new Error("JSON root must be an object.");

  const obj = raw as Record<string, unknown>;
  const contours = obj.contours;
  if (!Array.isArray(contours)) throw new Error("Missing `contours` array.");

  const parsedContours: KkgContour[] = contours.map((c, idx) => {
    if (!c || typeof c !== "object") {
      throw new Error(`contours[${idx}] must be an object.`);
    }
    const co = c as Record<string, unknown>;

    const id = co.id;
    const labelId = co.label_id;
    const vertices = co.vertices;
    if (typeof id !== "number") throw new Error(`contours[${idx}].id must be a number.`);
    if (typeof labelId !== "number")
      throw new Error(`contours[${idx}].label_id must be a number.`);
    if (!Array.isArray(vertices))
      throw new Error(`contours[${idx}].vertices must be an array.`);

    const parsedVertices: KkgVertex[] = vertices.map((v, j) => {
      if (!v || typeof v !== "object") {
        throw new Error(`contours[${idx}].vertices[${j}] must be an object.`);
      }
      const vo = v as Record<string, unknown>;
      const x = vo.x;
      const y = vo.y;
      if (typeof x !== "number" || typeof y !== "number") {
        throw new Error(
          `contours[${idx}].vertices[${j}] must have numeric x/y.`,
        );
      }
      return { x, y };
    });

    return {
      id,
      label_id: labelId,
      vertices: parsedVertices,
      color: typeof co.color === "string" ? co.color : null,
      room_name: typeof co.room_name === "string" ? co.room_name : null,
      area: typeof co.area === "number" ? co.area : undefined,
      perimeter: typeof co.perimeter === "number" ? co.perimeter : undefined,
      holes: co.holes,
      isLine: typeof co.isLine === "boolean" ? co.isLine : null,
      lineWidth: typeof co.lineWidth === "number" ? co.lineWidth : null,
      instanceGroupId: co.instanceGroupId,
    };
  });

  const imageDataRaw = obj.image_data;
  let image_data: KkgJson["image_data"] = null;
  if (imageDataRaw && typeof imageDataRaw === "object") {
    const ido = imageDataRaw as Record<string, unknown>;
    if (typeof ido.base64 === "string" && typeof ido.mime_type === "string") {
      image_data = { base64: ido.base64, mime_type: ido.mime_type };
    }
  }

  return {
    contours: parsedContours,
    image_data,
    image_width: typeof obj.image_width === "number" ? obj.image_width : undefined,
    image_height: typeof obj.image_height === "number" ? obj.image_height : undefined,
    timestamp: typeof obj.timestamp === "string" ? obj.timestamp : undefined,
  };
}

