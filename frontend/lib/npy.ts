const MAGIC = "\u0093NUMPY";
const BEAT_LENGTH = 187;
const MIT_BIH_FS = 360;

type ParsedNpy = {
  shape: number[];
  data: number[];
};

function parseHeader(headerText: string): { descr: string; fortran_order: boolean; shape: number[] } {
  const descr = /'descr':\s*'([^']+)'/.exec(headerText)?.[1];
  const fortranMatch = /'fortran_order':\s*(True|False)/.exec(headerText)?.[1];
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(headerText)?.[1];

  if (!descr || !fortranMatch || !shapeText) {
    throw new Error("Unsupported .npy header.");
  }

  const shape = shapeText
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number.parseInt(item, 10));

  return {
    descr,
    fortran_order: fortranMatch === "True",
    shape,
  };
}

function product(values: number[]) {
  return values.reduce((acc, value) => acc * value, 1);
}

function bytesPerElement(dtype: string) {
  return Number.parseInt(dtype.slice(2), 10);
}

function parseTypedData(view: DataView, offset: number, dtype: string, count: number): number[] {
  const values: number[] = [];
  const littleEndian = dtype.startsWith("<") || dtype.startsWith("|");
  const kind = dtype.slice(1);
  const step = bytesPerElement(dtype);

  for (let index = 0; index < count; index += 1) {
    const position = offset + index * step;

    switch (kind) {
      case "f4":
        values.push(view.getFloat32(position, littleEndian));
        break;
      case "f8":
        values.push(view.getFloat64(position, littleEndian));
        break;
      case "i4":
        values.push(view.getInt32(position, littleEndian));
        break;
      case "i2":
        values.push(view.getInt16(position, littleEndian));
        break;
      case "u1":
        values.push(view.getUint8(position));
        break;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
  }

  return values;
}

export async function parseNpyFile(file: File): Promise<ParsedNpy> {
  const buffer = await file.arrayBuffer();
  const view = new DataView(buffer);

  const magic = String.fromCharCode(...new Uint8Array(buffer.slice(0, 6)));
  if (magic !== MAGIC) {
    throw new Error("Invalid .npy file.");
  }

  const major = view.getUint8(6);
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const headerText = new TextDecoder("latin1")
    .decode(buffer.slice(headerOffset, headerOffset + headerLength))
    .trim();

  const header = parseHeader(headerText);
  if (header.fortran_order) {
    throw new Error("Fortran-ordered .npy arrays are not supported yet.");
  }

  const count = product(header.shape);
  const dataOffset = headerOffset + headerLength;
  const data = parseTypedData(view, dataOffset, header.descr, count);

  return { shape: header.shape, data };
}

export function toPredictPayload(patientId: string, parsed: ParsedNpy) {
  if (parsed.shape.length !== 2 || parsed.shape[1] !== BEAT_LENGTH) {
    throw new Error("Expected a 2D beat array with shape (N, 187).");
  }

  const beatCount = parsed.shape[0];
  const beats: number[][] = [];

  for (let row = 0; row < beatCount; row += 1) {
    const start = row * BEAT_LENGTH;
    beats.push(parsed.data.slice(start, start + BEAT_LENGTH));
  }

  const timestamps = Array.from(
    { length: beatCount },
    (_, index) => index * (BEAT_LENGTH / MIT_BIH_FS),
  );

  return {
    patient_id: patientId,
    beats,
    timestamps,
    fs: MIT_BIH_FS,
    save_session: true,
  };
}
