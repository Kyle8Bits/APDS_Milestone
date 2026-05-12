import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

function hashString(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash)
  }
  return hash
}

function brandGradient(brand) {
  const h = Math.abs(hashString(brand || 'default'))
  const hue1 = h % 360
  const hue2 = (hue1 + 40) % 360
  return `linear-gradient(135deg, hsl(${hue1}, 65%, 72%), hsl(${hue2}, 55%, 58%))`
}

function renderStars(rating) {
  const rounded = Math.round(Number(rating) || 0)
  const clamped = Math.max(0, Math.min(5, rounded))
  const filled = '★'.repeat(clamped)
  const empty = '☆'.repeat(5 - clamped)
  return { filled, empty }
}

function formatPrice(price) {
  return Number(price).toFixed(2)
}

function ProductCard({ product, compact }) {
  const {
    product_id,
    product_title,
    brand_name,
    price,
    avg_product_rating,
    review_count,
    image
  } = product

  const displayTitle = product_title && product_title.length > 60
    ? product_title.slice(0, 57) + '...'
    : product_title

  const stars = avg_product_rating != null ? renderStars(avg_product_rating) : null

  return (
    <motion.div
      whileHover={{ y: -4, scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
    >
      <Link
        to={`/products/${product_id}`}
        className={`block bg-surface rounded-xl shadow-sm overflow-hidden transition-shadow duration-200 hover:shadow-lg${compact ? ' min-w-[200px] max-w-[200px] shrink-0' : ''}`}
      >
        <div className={`relative rounded-t-xl ${compact ? 'h-[130px]' : 'h-[200px]'}`}>
          {image && (
            <img
              src={image}
              alt={product_title}
              className="absolute inset-0 h-full w-full object-cover"
            />
          )}
          <div
            className="absolute inset-0 opacity-20"
            style={{ background: brandGradient(brand_name) }}
          />
        </div>
        <div className={compact ? 'p-2.5' : 'p-4'}>
          <h3 className={`font-semibold text-primary mb-1.5 leading-snug ${compact ? 'text-[13px] min-h-0' : 'text-[15px] min-h-[40px]'}`}>{displayTitle}</h3>
          <p className="text-[13px] text-text-light mb-2.5">{brand_name}</p>
          <div className="flex items-center gap-3 flex-wrap">
            {price != null && (
              <span className="text-[17px] font-bold text-primary">${formatPrice(price)}</span>
            )}
            {stars && (
              <span className="text-sm text-text flex items-center gap-0.5">
                <span className="text-accent tracking-wide">{stars.filled}<span className="text-border-light">{stars.empty}</span></span>
                <span className="text-text-light text-[0.85rem] ml-1">{Number(avg_product_rating).toFixed(1)}</span>
              </span>
            )}
            {review_count != null && (
              <span className="text-xs text-text-light">{review_count} reviews</span>
            )}
          </div>
        </div>
      </Link>
    </motion.div>
  )
}

export { brandGradient, renderStars, formatPrice }
export default ProductCard
