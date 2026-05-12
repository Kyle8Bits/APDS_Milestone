import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import ProductCard from '../components/ProductCard'

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } }
}

const stagger = {
  visible: { transition: { staggerChildren: 0.08 } }
}

function SentimentBar({ positive, neutral, negative, total }) {
  if (!total) return null
  const pPct = (positive / total) * 100
  const neuPct = (neutral / total) * 100
  const nPct = (negative / total) * 100

  return (
    <div className="w-full">
      <div className="flex h-8 rounded-lg overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pPct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="bg-success flex items-center justify-center text-white text-xs font-semibold"
        >
          {pPct >= 8 && `${pPct.toFixed(0)}%`}
        </motion.div>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${neuPct}%` }}
          transition={{ duration: 0.8, delay: 0.1, ease: 'easeOut' }}
          className="bg-accent flex items-center justify-center text-white text-xs font-semibold"
        >
          {neuPct >= 8 && `${neuPct.toFixed(0)}%`}
        </motion.div>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${nPct}%` }}
          transition={{ duration: 0.8, delay: 0.2, ease: 'easeOut' }}
          className="bg-danger flex items-center justify-center text-white text-xs font-semibold"
        >
          {nPct >= 8 && `${nPct.toFixed(0)}%`}
        </motion.div>
      </div>
      <div className="flex justify-between mt-2 text-xs text-text-light">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-success inline-block" /> Positive ({positive.toLocaleString()})</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-accent inline-block" /> Neutral ({neutral.toLocaleString()})</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-danger inline-block" /> Negative ({negative.toLocaleString()})</span>
      </div>
    </div>
  )
}

function SentimentGauge({ value }) {
  const normalized = ((value + 1) / 2) * 100
  let color = 'text-accent'
  let label = 'Neutral'
  if (value >= 0.3) { color = 'text-success'; label = 'Positive' }
  else if (value >= 0.05) { color = 'text-success'; label = 'Slightly Positive' }
  else if (value <= -0.3) { color = 'text-danger'; label = 'Negative' }
  else if (value <= -0.05) { color = 'text-danger'; label = 'Slightly Negative' }

  return (
    <div className="flex flex-col items-center gap-2">
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.2 }}
        className={`text-[48px] font-bold ${color}`}
      >
        {(value * 100).toFixed(0)}
      </motion.div>
      <div className="w-full h-3 bg-bg-light rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${normalized}%` }}
          transition={{ duration: 1, ease: 'easeOut', delay: 0.3 }}
          className="h-full rounded-full bg-gradient-to-r from-danger via-accent to-success"
        />
      </div>
      <div className="flex justify-between w-full text-[11px] text-text-lighter">
        <span>-100</span>
        <span className={`font-semibold text-sm ${color}`}>{label}</span>
        <span>+100</span>
      </div>
    </div>
  )
}

function AnimatedCounter({ value, suffix = '' }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 200, damping: 15 }}
      className="text-[32px] font-bold text-primary mb-1"
    >
      {typeof value === 'number' ? value.toLocaleString() : value}{suffix}
    </motion.div>
  )
}

function Dashboard() {
  const [stats, setStats] = useState(null)
  const [topRated, setTopRated] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const carouselRef = useRef(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/stats').then(r => r.ok ? r.json() : Promise.reject('Failed to fetch stats')),
      fetch('/api/top-rated?n=10').then(r => r.ok ? r.json() : [])
    ])
      .then(([statsData, topData]) => {
        setStats(statsData)
        setTopRated(Array.isArray(topData) ? topData : [])
        setLoading(false)
      })
      .catch(err => {
        setError(typeof err === 'string' ? err : err.message)
        setLoading(false)
      })
  }, [])

  const scrollCarousel = (dir) => {
    if (!carouselRef.current) return
    carouselRef.current.scrollBy({ left: dir * 320, behavior: 'smooth' })
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-text-light">
        <div className="w-8 h-8 border-3 border-border-light border-t-secondary rounded-full animate-spin" />
        <span>Loading dashboard...</span>
      </div>
    )
  }

  if (error) {
    return <div className="text-center py-16 px-6 text-danger text-lg">{error}</div>
  }

  const buyerRate = stats.total_reviews > 0
    ? ((stats.buyer_vs_non_buyer.buyers / stats.total_reviews) * 100).toFixed(1)
    : '0.0'

  const maxRating = Math.max(...Object.values(stats.rating_distribution), 1)
  const sentiment = stats.sentiment || {}

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-8 pb-16">
      <motion.div initial="hidden" animate="visible" variants={stagger}>
        {/* Header */}
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="text-[28px] font-bold text-primary mb-1.5">Dashboard</h1>
          <p className="text-[15px] text-text-light">Product insights, review analytics & sentiment analysis</p>
        </motion.div>

        {/* Top Rated Carousel */}
        {topRated.length > 0 && (
          <motion.section variants={fadeUp} className="mb-10">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-primary">Top Rated Products</h2>
              <div className="flex gap-2">
                <button
                  onClick={() => scrollCarousel(-1)}
                  className="w-9 h-9 flex items-center justify-center rounded-full border border-border bg-surface text-primary hover:bg-secondary hover:text-white hover:border-secondary transition"
                  aria-label="Scroll left"
                >&#8249;</button>
                <button
                  onClick={() => scrollCarousel(1)}
                  className="w-9 h-9 flex items-center justify-center rounded-full border border-border bg-surface text-primary hover:bg-secondary hover:text-white hover:border-secondary transition"
                  aria-label="Scroll right"
                >&#8250;</button>
              </div>
            </div>
            <div ref={carouselRef} className="flex gap-4 overflow-x-auto pb-3 scrollbar-thin scroll-smooth">
              {topRated.map((p) => (
                <ProductCard key={p.product_id} product={p} compact />
              ))}
            </div>
          </motion.section>
        )}

        {/* Summary Cards */}
        <motion.div variants={fadeUp} className="grid grid-cols-2 md:grid-cols-4 gap-5 mb-10">
          {[
            { value: stats.total_products, label: 'Products' },
            { value: stats.total_reviews, label: 'Reviews' },
            { value: stats.total_brands, label: 'Brands' },
            { value: buyerRate, label: 'Buyer Rate', suffix: '%' },
          ].map((card, i) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.08, duration: 0.4 }}
              className="bg-surface rounded-xl p-6 shadow-sm text-center border-t-3 border-secondary"
            >
              <AnimatedCounter value={card.value} suffix={card.suffix} />
              <div className="text-[13px] font-semibold text-text-light uppercase tracking-wide">{card.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Sentiment Analysis Section */}
        <motion.div variants={fadeUp} className="mb-10">
          <h2 className="text-xl font-bold text-primary mb-4">Sentiment Analysis</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="bg-surface rounded-xl p-6 shadow-sm"
            >
              <h3 className="text-[15px] font-semibold text-primary mb-4">Overall Sentiment Score</h3>
              <SentimentGauge value={sentiment.avg_compound || 0} />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4, duration: 0.5 }}
              className="bg-surface rounded-xl p-6 shadow-sm"
            >
              <h3 className="text-[15px] font-semibold text-primary mb-4">Sentiment Distribution</h3>
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="p-3 rounded-lg bg-success/10">
                    <div className="text-2xl font-bold text-success">{(sentiment.positive || 0).toLocaleString()}</div>
                    <div className="text-xs text-text-light mt-1">Positive</div>
                  </div>
                  <div className="p-3 rounded-lg bg-accent/10">
                    <div className="text-2xl font-bold text-accent">{(sentiment.neutral || 0).toLocaleString()}</div>
                    <div className="text-xs text-text-light mt-1">Neutral</div>
                  </div>
                  <div className="p-3 rounded-lg bg-danger/10">
                    <div className="text-2xl font-bold text-danger">{(sentiment.negative || 0).toLocaleString()}</div>
                    <div className="text-xs text-text-light mt-1">Negative</div>
                  </div>
                </div>
                <SentimentBar
                  positive={sentiment.positive || 0}
                  neutral={sentiment.neutral || 0}
                  negative={sentiment.negative || 0}
                  total={sentiment.total || 0}
                />
              </div>
            </motion.div>
          </div>
        </motion.div>

        {/* Sentiment by Brand */}
        {sentiment.by_brand && sentiment.by_brand.length > 0 && (
          <motion.div variants={fadeUp} className="mb-10">
            <h2 className="text-xl font-bold text-primary mb-4">Sentiment by Brand</h2>
            <div className="bg-surface rounded-xl shadow-sm overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead className="bg-primary">
                  <tr>
                    <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Brand</th>
                    <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Avg Score</th>
                    <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Positive</th>
                    <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Neutral</th>
                    <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Negative</th>
                    <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Sentiment</th>
                  </tr>
                </thead>
                <tbody>
                  {sentiment.by_brand.map((b, index) => {
                    const pct = b.total ? (b.positive / b.total) * 100 : 0
                    let scoreColor = 'text-accent'
                    if (b.avg_sentiment >= 0.05) scoreColor = 'text-success'
                    else if (b.avg_sentiment <= -0.05) scoreColor = 'text-danger'

                    return (
                      <motion.tr
                        key={b.brand}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.4 + index * 0.04, duration: 0.3 }}
                        className="hover:bg-surface-hover"
                      >
                        <td className={`px-4 py-3 text-text font-medium ${index < sentiment.by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{b.brand}</td>
                        <td className={`px-4 py-3 font-semibold ${scoreColor} ${index < sentiment.by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{(b.avg_sentiment * 100).toFixed(0)}</td>
                        <td className={`px-4 py-3 text-text ${index < sentiment.by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{b.positive.toLocaleString()}</td>
                        <td className={`px-4 py-3 text-text ${index < sentiment.by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{b.neutral.toLocaleString()}</td>
                        <td className={`px-4 py-3 text-text ${index < sentiment.by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{b.negative.toLocaleString()}</td>
                        <td className={`px-4 py-3 ${index < sentiment.by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>
                          <div className="flex h-4 rounded-full overflow-hidden min-w-[100px]">
                            <div className="bg-success" style={{ width: `${pct}%` }} />
                            <div className="bg-accent" style={{ width: `${b.total ? (b.neutral / b.total) * 100 : 0}%` }} />
                            <div className="bg-danger" style={{ width: `${b.total ? (b.negative / b.total) * 100 : 0}%` }} />
                          </div>
                        </td>
                      </motion.tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}

        {/* Rating Distribution */}
        <motion.div variants={fadeUp} className="mb-10">
          <h2 className="text-xl font-bold text-primary mb-4">Rating Distribution</h2>
          <div className="bg-surface rounded-xl p-6 shadow-sm">
            {['5', '4', '3', '2', '1'].map((star, index) => (
              <div className={`flex items-center gap-3 ${index < 4 ? 'mb-3' : ''}`} key={star}>
                <span className="text-sm font-semibold text-text min-w-14">{star} star</span>
                <div className="flex-1 h-6 bg-bg-light rounded-lg overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(stats.rating_distribution[star] / maxRating) * 100}%` }}
                    transition={{ duration: 0.7, delay: 0.5 + index * 0.08, ease: 'easeOut' }}
                    className="h-full rounded-lg min-w-[2px] bg-gradient-to-r from-secondary to-secondary-light"
                  />
                </div>
                <span className="text-sm font-semibold text-text-light min-w-[60px] text-right">{stats.rating_distribution[star].toLocaleString()}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Brand Breakdown Table */}
        <motion.div variants={fadeUp} className="mb-10">
          <h2 className="text-xl font-bold text-primary mb-4">Brand Breakdown</h2>
          <div className="bg-surface rounded-xl shadow-sm overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead className="bg-primary">
                <tr>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Brand</th>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Reviews</th>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Buyer %</th>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Avg Rating</th>
                </tr>
              </thead>
              <tbody>
                {stats.reviews_by_brand.map((b, index) => (
                  <tr key={b.brand} className="hover:bg-surface-hover">
                    <td className={`px-4 py-3 text-text ${index < stats.reviews_by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{b.brand}</td>
                    <td className={`px-4 py-3 text-text ${index < stats.reviews_by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{b.reviews.toLocaleString()}</td>
                    <td className={`px-4 py-3 text-text ${index < stats.reviews_by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{b.buyer_pct}%</td>
                    <td className={`px-4 py-3 text-text ${index < stats.reviews_by_brand.length - 1 ? 'border-b border-border-light' : ''}`}>{b.avg_rating}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Top Products Table */}
        <motion.div variants={fadeUp} className="mb-10">
          <h2 className="text-xl font-bold text-primary mb-4">Top 5 Most Reviewed Products</h2>
          <div className="bg-surface rounded-xl shadow-sm overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead className="bg-primary">
                <tr>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Name</th>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Brand</th>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Reviews</th>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Buyer %</th>
                  <th className="px-4 py-3.5 text-left font-semibold text-white text-[13px] uppercase tracking-wide">Avg Rating</th>
                </tr>
              </thead>
              <tbody>
                {stats.top_products.map((p, index) => (
                  <tr key={p.product_id} className="hover:bg-surface-hover">
                    <td className={`px-4 py-3 text-text max-w-[300px] whitespace-nowrap overflow-hidden text-ellipsis ${index < stats.top_products.length - 1 ? 'border-b border-border-light' : ''}`}>{p.product_title}</td>
                    <td className={`px-4 py-3 text-text ${index < stats.top_products.length - 1 ? 'border-b border-border-light' : ''}`}>{p.brand_name}</td>
                    <td className={`px-4 py-3 text-text ${index < stats.top_products.length - 1 ? 'border-b border-border-light' : ''}`}>{p.review_count.toLocaleString()}</td>
                    <td className={`px-4 py-3 text-text ${index < stats.top_products.length - 1 ? 'border-b border-border-light' : ''}`}>{p.buyer_pct}%</td>
                    <td className={`px-4 py-3 text-text ${index < stats.top_products.length - 1 ? 'border-b border-border-light' : ''}`}>{p.avg_rating}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default Dashboard
