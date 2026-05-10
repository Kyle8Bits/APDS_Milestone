import { useState, useEffect } from 'react'

function Dashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/stats')
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch stats')
        return res.json()
      })
      .then(data => {
        setStats(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

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

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-8 pb-16">
      <div className="mb-6">
        <h1 className="text-[28px] font-bold text-primary mb-1.5">Review Statistics</h1>
        <p className="text-[15px] text-text-light">Overview of product reviews and buyer insights</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-5 mb-10">
        <div className="bg-surface rounded-xl p-6 shadow-sm text-center border-t-3 border-secondary">
          <div className="text-[32px] font-bold text-primary mb-1">{stats.total_products.toLocaleString()}</div>
          <div className="text-[13px] font-semibold text-text-light uppercase tracking-wide">Total Products</div>
        </div>
        <div className="bg-surface rounded-xl p-6 shadow-sm text-center border-t-3 border-secondary">
          <div className="text-[32px] font-bold text-primary mb-1">{stats.total_reviews.toLocaleString()}</div>
          <div className="text-[13px] font-semibold text-text-light uppercase tracking-wide">Total Reviews</div>
        </div>
        <div className="bg-surface rounded-xl p-6 shadow-sm text-center border-t-3 border-secondary">
          <div className="text-[32px] font-bold text-primary mb-1">{stats.total_brands}</div>
          <div className="text-[13px] font-semibold text-text-light uppercase tracking-wide">Total Brands</div>
        </div>
        <div className="bg-surface rounded-xl p-6 shadow-sm text-center border-t-3 border-secondary">
          <div className="text-[32px] font-bold text-primary mb-1">{buyerRate}%</div>
          <div className="text-[13px] font-semibold text-text-light uppercase tracking-wide">Buyer Rate</div>
        </div>
      </div>

      {/* Rating Distribution */}
      <div className="mb-10">
        <h2 className="text-xl font-bold text-primary mb-4">Rating Distribution</h2>
        <div className="bg-surface rounded-xl p-6 shadow-sm">
          {['5', '4', '3', '2', '1'].map((star, index) => (
            <div className={`flex items-center gap-3 ${index < 4 ? 'mb-3' : ''}`} key={star}>
              <span className="text-sm font-semibold text-text min-w-14">{star} star</span>
              <div className="flex-1 h-6 bg-bg-light rounded-lg overflow-hidden">
                <div
                  className="h-full rounded-lg min-w-[2px] bg-gradient-to-r from-secondary to-secondary-light"
                  style={{ width: `${(stats.rating_distribution[star] / maxRating) * 100}%` }}
                />
              </div>
              <span className="text-sm font-semibold text-text-light min-w-[60px] text-right">{stats.rating_distribution[star].toLocaleString()}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Brand Breakdown Table */}
      <div className="mb-10">
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
      </div>

      {/* Top Products Table */}
      <div className="mb-10">
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
      </div>
    </div>
  )
}

export default Dashboard
