import { useState } from "react";
import { api, type Product, type SearchResponse } from "../api";
import { ProductCard } from "./ProductCard";

const EXAMPLES = ["navy formal shirt", "red party dress", "white sneakers", "ethnic kurta", "black leather belt"];
const COLORS = ["", "Black", "Blue", "Navy Blue", "White", "Grey", "Green", "Red", "Pink", "Yellow", "Purple", "Brown"];

export function SearchView({ onSelect }: { onSelect: (p: Product) => void }) {
  const [q, setQ] = useState("");
  const [gender, setGender] = useState("");
  const [color, setColor] = useState("");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(text = q, offset = 0) {
    if (!text.trim()) return;
    setQ(text);
    setBusy(true);
    setError(null);
    try {
      const page = await api.search(text, gender, color, offset);
      setResult((prev) => (offset && prev ? { ...page, items: [...prev.items, ...page.items] } : page));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card">
      <h2>Search the catalog</h2>
      <p className="muted">
        Hybrid retrieval: BM25 keyword matching fused with CLIP semantic similarity using reciprocal rank fusion.
      </p>
      <form
        className="search-bar"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Describe what you're looking for…" />
        <select value={gender} onChange={(e) => setGender(e.target.value)} aria-label="Gender">
          <option value="">Anyone</option>
          <option>Men</option>
          <option>Women</option>
          <option>Boys</option>
          <option>Girls</option>
        </select>
        <select value={color} onChange={(e) => setColor(e.target.value)} aria-label="Colour">
          {COLORS.map((c) => (
            <option key={c} value={c}>
              {c || "Any colour"}
            </option>
          ))}
        </select>
        <button className="primary" disabled={busy}>
          Search
        </button>
      </form>
      <div className="chips">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => run(ex)}>
            {ex}
          </button>
        ))}
      </div>

      {error && <div className="alert error">{error}</div>}
      {result && (
        <>
          <div className="result-meta muted small">
            {result.total} matches · mode: <code>{result.mode}</code>
            {result.detected_colors.length > 0 && <> · colour filter from your query: {result.detected_colors.join(", ")}</>}
          </div>
          {result.items.length === 0 ? (
            <div className="empty">No products matched. Try different words.</div>
          ) : (
            <div className="product-grid">
              {result.items.map((hit) => (
                <ProductCard
                  key={hit.product.id}
                  product={hit.product}
                  onSelect={onSelect}
                  badge={
                    <span className="rank-badges">
                      {hit.sparse_rank && <span title="BM25 rank">BM25 #{hit.sparse_rank}</span>}
                      {hit.dense_rank && <span title="CLIP rank">CLIP #{hit.dense_rank}</span>}
                    </span>
                  }
                />
              ))}
            </div>
          )}
          {result.items.length < result.total && (
            <button className="secondary load-more" disabled={busy} onClick={() => run(result.query, result.items.length)}>
              Load more
            </button>
          )}
        </>
      )}
    </section>
  );
}
