import { useCallback, useEffect, useState } from "react";
import { api, type Health, type Preferences, type Product, type Recommendations, type SkinAnalysis, type Tone } from "./api";
import { ProductModal } from "./components/ProductModal";
import { PreferenceBar, RecommendationResults } from "./components/Recommendations";
import { SearchView } from "./components/SearchView";
import { ToneFinder } from "./components/ToneFinder";

type View = "discover" | "search";

export default function App() {
  const [view, setView] = useState<View>("discover");
  const [health, setHealth] = useState<Health | null>(null);
  const [tones, setTones] = useState<Tone[]>([]);
  const [toneId, setToneId] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<SkinAnalysis | null>(null);
  const [prefs, setPrefs] = useState<Preferences>({ gender: "Women", category: "Apparel", usage: "", query: "" });
  const [seed, setSeed] = useState(0);
  const [recs, setRecs] = useState<Recommendations | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Product | null>(null);

  useEffect(() => {
    Promise.all([api.health(), api.tones()])
      .then(([h, t]) => {
        setHealth(h);
        setTones(t);
      })
      .catch(() => setError("Can't reach the HueMatch API. Is the backend running on port 8000?"));
  }, []);

  useEffect(() => {
    if (!toneId) return;
    let live = true;
    setLoading(true);
    api
      .recommendations(toneId, prefs, seed)
      .then((r) => {
        if (!live) return;
        setRecs(r);
        setError(null);
      })
      .catch((e: Error) => live && setError(e.message))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [toneId, prefs, seed]);

  const closeModal = useCallback(() => setSelected(null), []);

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
            <circle cx="11" cy="16" r="9" fill="#E2C2AD" />
            <circle cx="21" cy="16" r="9" fill="#6A4D3F" fillOpacity=".85" />
          </svg>
          HueMatch
        </div>
        <nav className="tabs">
          <button className={view === "discover" ? "active" : ""} onClick={() => setView("discover")}>
            Discover
          </button>
          <button className={view === "search" ? "active" : ""} onClick={() => setView("search")}>
            Search
          </button>
        </nav>
        {health && (
          <span className="status small muted" title={`Search mode: ${health.search_mode}`}>
            <span className="status-dot" /> {health.products.toLocaleString()} products
          </span>
        )}
      </header>

      <main className="container">
        {view === "discover" ? (
          <>
            <section className="hero">
              <h1>Wear the colours that love you back.</h1>
              <p className="muted">
                Upload a selfie and HueMatch measures your skin tone, builds a flattering colour palette and picks
                products from a catalog of 20,000+ items to match.
              </p>
            </section>
            {error && <div className="alert error">{error}</div>}
            <ToneFinder
              tones={tones}
              selected={toneId}
              analysis={analysis}
              onAnalysis={setAnalysis}
              onSelect={setToneId}
            />
            {toneId && <PreferenceBar prefs={prefs} onChange={setPrefs} onShuffle={() => setSeed((s) => s + 1)} />}
            {recs && toneId && <RecommendationResults recs={recs} loading={loading} onSelect={setSelected} />}
          </>
        ) : (
          <SearchView onSelect={setSelected} />
        )}
      </main>

      <footer className="footer muted small">
        Product data from the Fashion Product Images dataset (Kaggle). Skin tone estimates use the Individual
        Typology Angle and are a styling aid, not a medical assessment.
      </footer>

      {selected && <ProductModal product={selected} onClose={closeModal} onSelect={setSelected} />}
    </>
  );
}
