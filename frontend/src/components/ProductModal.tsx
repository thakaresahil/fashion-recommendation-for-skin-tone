import { useEffect, useState } from "react";
import { api, type Product } from "../api";
import { ProductCard, ProductImage } from "./ProductCard";

export function ProductModal({
  product,
  onClose,
  onSelect,
}: {
  product: Product;
  onClose: () => void;
  onSelect: (p: Product) => void;
}) {
  const [similar, setSimilar] = useState<Product[] | null>(null);

  useEffect(() => {
    let live = true;
    setSimilar(null);
    api
      .similar(product.id)
      .then((items) => live && setSimilar(items))
      .catch(() => live && setSimilar([]));
    return () => {
      live = false;
    };
  }, [product.id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={product.name} onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <div className="modal-main">
          <div className="modal-media">
            <ProductImage key={product.id} product={product} large />
          </div>
          <div className="modal-info">
            <h3>{product.name}</h3>
            <dl>
              <dt>Colour</dt>
              <dd>
                <span className="dot" style={{ background: product.color_hex }} /> {product.color}
              </dd>
              <dt>Type</dt>
              <dd>{product.article_type}</dd>
              <dt>Category</dt>
              <dd>
                {product.master_category} / {product.sub_category}
              </dd>
              <dt>For</dt>
              <dd>{product.gender}</dd>
              {product.usage && (
                <>
                  <dt>Occasion</dt>
                  <dd>{product.usage}</dd>
                </>
              )}
              <dt>Product ID</dt>
              <dd>#{product.id}</dd>
            </dl>
          </div>
        </div>
        <h4>Similar items</h4>
        {similar === null ? (
          <div className="muted small">Finding similar items…</div>
        ) : (
          <div className="product-grid compact">
            {similar.map((p) => (
              <ProductCard key={p.id} product={p} onSelect={onSelect} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
