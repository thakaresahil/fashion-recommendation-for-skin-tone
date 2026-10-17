import type { Preferences, Product, Recommendations as Recs } from "../api";
import { ProductCard } from "./ProductCard";

const CATEGORIES = [
  { value: "", label: "All categories" },
  { value: "Apparel", label: "Clothing" },
  { value: "Footwear", label: "Footwear" },
  { value: "Accessories", label: "Accessories" },
];
const OCCASIONS = ["", "Casual", "Formal", "Sports", "Ethnic", "Party", "Smart Casual"];
const GENDERS = ["Women", "Men", "Girls", "Boys"];

export function PreferenceBar({
  prefs,
  onChange,
  onShuffle,
}: {
  prefs: Preferences;
  onChange: (p: Preferences) => void;
  onShuffle: () => void;
}) {
  return (
    <section className="card prefs">
      <div className="step-label">Step 2</div>
      <h2>Tell us what you're shopping for</h2>
      <div className="prefs-row">
        <div className="segmented" role="radiogroup" aria-label="Shopping for">
          {GENDERS.map((g) => (
            <button
              key={g}
              role="radio"
              aria-checked={prefs.gender === g}
              className={prefs.gender === g ? "active" : ""}
              onClick={() => onChange({ ...prefs, gender: g })}
            >
              {g}
            </button>
          ))}
        </div>
        <select
          value={prefs.category}
          onChange={(e) => onChange({ ...prefs, category: e.target.value })}
          aria-label="Category"
        >
          {CATEGORIES.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
        <select value={prefs.usage} onChange={(e) => onChange({ ...prefs, usage: e.target.value })} aria-label="Occasion">
          {OCCASIONS.map((o) => (
            <option key={o} value={o}>
              {o || "Any occasion"}
            </option>
          ))}
        </select>
        <form
          className="refine"
          onSubmit={(e) => {
            e.preventDefault();
            const value = new FormData(e.currentTarget).get("q");
            onChange({ ...prefs, query: String(value ?? "") });
          }}
        >
          <input name="q" defaultValue={prefs.query} placeholder="Refine, e.g. “cotton kurta”" />
        </form>
        <button className="secondary" onClick={onShuffle} title="Show different products">
          ↻ Shuffle
        </button>
      </div>
    </section>
  );
}

export function RecommendationResults({
  recs,
  loading,
  onSelect,
}: {
  recs: Recs;
  loading: boolean;
  onSelect: (p: Product) => void;
}) {
  const nonEmpty = recs.groups.filter((g) => g.items.length > 0);
  return (
    <section className={`results${loading ? " loading" : ""}`}>
      <div className="palette-header">
        <span className="swatch-circle big" style={{ background: recs.tone.swatch }} />
        <div>
          <h2>Your palette: {recs.tone.label}</h2>
          <p className="muted">{recs.tone.description}</p>
        </div>
      </div>
      <nav className="palette-strip" aria-label="Palette colours">
        {recs.groups.map((g) => (
          <a key={g.color.name} href={`#color-${g.color.name.replace(/\s/g, "-")}`} className="palette-chip">
            <span className="palette-chip-swatch" style={{ background: g.color.hex }} />
            {g.color.name}
            <span className="muted small">{g.total}</span>
          </a>
        ))}
      </nav>
      {nonEmpty.length === 0 && <div className="empty">No products match these filters. Try widening them.</div>}
      {nonEmpty.map((g) => (
        <div key={g.color.name} id={`color-${g.color.name.replace(/\s/g, "-")}`} className="color-section">
          <h3>
            <span className="palette-chip-swatch" style={{ background: g.color.hex }} />
            {g.color.name}
            <span className="muted small">
              showing {g.items.length} of {g.total}
            </span>
          </h3>
          <div className="product-grid">
            {g.items.map((p) => (
              <ProductCard key={p.id} product={p} onSelect={onSelect} />
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
