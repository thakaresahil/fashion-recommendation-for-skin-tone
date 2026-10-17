import { useRef, useState } from "react";
import { api, type SkinAnalysis, type Tone } from "../api";

interface Props {
  tones: Tone[];
  selected: string | null;
  analysis: SkinAnalysis | null;
  onAnalysis: (a: SkinAnalysis | null) => void;
  onSelect: (toneId: string) => void;
}

export function ToneFinder({ tones, selected, analysis, onAnalysis, onSelect }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  async function handleFile(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    setError(null);
    setBusy(true);
    try {
      const result = await api.analyze(file);
      onAnalysis(result);
      onSelect(result.tone.id);
    } catch (e) {
      onAnalysis(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card tone-finder">
      <div className="step-label">Step 1</div>
      <h2>Find your skin tone</h2>
      <div className="finder-grid">
        <div
          className={`dropzone${dragging ? " dragging" : ""}`}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFile(e.dataTransfer.files[0]);
          }}
          role="button"
          tabIndex={0}
        >
          {preview ? (
            <img src={preview} alt="Your upload" className="preview" />
          ) : (
            <div className="dropzone-empty">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M3 16.5V19a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-2.5M7.5 8.25 12 3.75l4.5 4.5M12 3.75V16" />
              </svg>
              <strong>Upload a selfie</strong>
              <span>Front-facing, natural light, no filter</span>
            </div>
          )}
          {busy && <div className="dropzone-busy">Analysing…</div>}
          <input
            ref={input}
            type="file"
            accept="image/*"
            capture="user"
            hidden
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        <div className="finder-side">
          {error && <div className="alert error">{error}</div>}
          {analysis && !error && <AnalysisSummary analysis={analysis} onSelect={onSelect} />}
          <p className="muted small">Your photo is analysed in memory and never stored. Or pick your tone directly:</p>
          <div className="tone-swatches">
            {tones
              .filter((t) => t.ita_range)
              .map((t) => (
                <button
                  key={t.id}
                  className={`tone-swatch${selected === t.id ? " active" : ""}`}
                  onClick={() => onSelect(t.id)}
                  aria-pressed={selected === t.id}
                >
                  <span className="swatch-circle" style={{ background: t.swatch }} />
                  <span>{t.label}</span>
                </button>
              ))}
          </div>
          <button
            className={`link-button${selected === "common" ? " active" : ""}`}
            onClick={() => onSelect("common")}
          >
            Not sure? Browse everyday essentials →
          </button>
        </div>
      </div>
    </section>
  );
}

function AnalysisSummary({ analysis, onSelect }: { analysis: SkinAnalysis; onSelect: (id: string) => void }) {
  const pct = Math.round(analysis.confidence * 100);
  return (
    <div className="analysis">
      <div className="analysis-head">
        <span className="swatch-circle big" style={{ background: analysis.skin_hex }} title="Measured skin colour" />
        <div>
          <div className="muted small">Detected tone</div>
          <div className="analysis-tone">{analysis.tone.label}</div>
        </div>
        <div className="confidence" title="Estimate confidence">
          <div className="confidence-bar">
            <span style={{ width: `${pct}%` }} />
          </div>
          <span className="small muted">{pct}% confidence</span>
        </div>
      </div>
      <div className="analysis-metrics small muted">
        ITA° {analysis.ita.toFixed(1)} · L* {analysis.lab.l.toFixed(0)} · a* {analysis.lab.a.toFixed(0)} · b*{" "}
        {analysis.lab.b.toFixed(0)} · {analysis.face_detected ? "face detected" : "no face detected"}
      </div>
      {analysis.alternative && (
        <button className="link-button" onClick={() => onSelect(analysis.alternative!.id)}>
          Close to the boundary. Try {analysis.alternative.label} too →
        </button>
      )}
      {analysis.notes.map((n) => (
        <div key={n} className="alert warn small">
          {n}
        </div>
      ))}
    </div>
  );
}
