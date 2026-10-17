export interface ColorOut {
  name: string;
  hex: string;
}

export interface Tone {
  id: string;
  label: string;
  swatch: string;
  description: string;
  colors: ColorOut[];
  ita_range: [number, number] | null;
}

export interface Product {
  id: number;
  name: string;
  gender: string;
  master_category: string;
  sub_category: string;
  article_type: string;
  color: string;
  color_hex: string;
  usage: string;
  year: number;
  image_url: string | null;
}

export interface ColorGroup {
  color: ColorOut;
  total: number;
  items: Product[];
}

export interface Recommendations {
  tone: Tone;
  groups: ColorGroup[];
}

export interface SkinAnalysis {
  tone: Tone;
  alternative: Tone | null;
  ita: number;
  lab: { l: number; a: number; b: number };
  skin_hex: string;
  face_detected: boolean;
  confidence: number;
  notes: string[];
}

export interface SearchHit {
  product: Product;
  score: number;
  sparse_rank: number | null;
  dense_rank: number | null;
}

export interface SearchResponse {
  query: string;
  mode: string;
  detected_colors: string[];
  total: number;
  items: SearchHit[];
}

export interface Health {
  status: string;
  products: number;
  images: number;
  search_mode: string;
}

export interface Preferences {
  gender: string;
  category: string;
  usage: string;
  query: string;
}

const BASE = import.meta.env.VITE_API_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}/api/v1${path}`, init);
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* keep the generic message */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

function query(params: Record<string, string | number | undefined>): string {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") qs.set(key, String(value));
  }
  return qs.toString();
}

export const api = {
  health: () => request<Health>("/health"),
  tones: () => request<Tone[]>("/skin-tones"),
  analyze: (file: File) => {
    const body = new FormData();
    body.append("image", file);
    return request<SkinAnalysis>("/skin-tone/analyze", { method: "POST", body });
  },
  recommendations: (tone: string, prefs: Preferences, seed: number) =>
    request<Recommendations>(
      `/recommendations?${query({
        skin_tone: tone,
        gender: prefs.gender,
        category: prefs.category,
        usage: prefs.usage,
        q: prefs.query.trim(),
        per_color: 12,
        seed,
      })}`,
    ),
  search: (q: string, gender: string, color: string, offset = 0) =>
    request<SearchResponse>(`/products/search?${query({ q, gender, color, limit: 24, offset })}`),
  similar: (id: number) => request<Product[]>(`/products/${id}/similar?limit=8`),
};
