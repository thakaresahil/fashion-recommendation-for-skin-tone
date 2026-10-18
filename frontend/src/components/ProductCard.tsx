import { useState } from "react";
import type { Product } from "../api";

function isLight(hex: string): boolean {
  const n = parseInt(hex.slice(1), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  return 0.299 * r + 0.587 * g + 0.114 * b > 160;
}

export function ProductImage({ product, large = false }: { product: Product; large?: boolean }) {
  const [failed, setFailed] = useState(false);
  if (product.image_url && !failed) {
    return (
      <img
        className="product-img"
        src={product.image_url}
        alt={product.name}
        loading="lazy"
        onError={() => setFailed(true)}
      />
    );
  }
  // Without a photo (images not downloaded, or it failed to load), show a colour tile instead.
  return (
    <div
      className={`product-placeholder${large ? " large" : ""}`}
      style={{
        background: `linear-gradient(150deg, ${product.color_hex}, ${product.color_hex}cc)`,
        color: isLight(product.color_hex) ? "#1d1d1f" : "#fff",
      }}
      role="img"
      aria-label={product.name}
    >
      <span className="placeholder-type">{product.article_type}</span>
      <span className="placeholder-color">{product.color}</span>
    </div>
  );
}

export function ProductCard({
  product,
  onSelect,
  badge,
}: {
  product: Product;
  onSelect: (p: Product) => void;
  badge?: React.ReactNode;
}) {
  return (
    <button className="product-card" onClick={() => onSelect(product)} title={product.name}>
      <div className="product-media">
        <ProductImage product={product} />
        {badge}
      </div>
      <div className="product-body">
        <span className="product-name">{product.name}</span>
        <span className="product-meta">
          <span className="dot" style={{ background: product.color_hex }} />
          {product.color} · {product.usage || product.sub_category}
        </span>
      </div>
    </button>
  );
}
