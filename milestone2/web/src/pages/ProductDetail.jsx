import { useState, useEffect, useRef } from 'react'
import { useParams } from 'react-router-dom'
import ProductCard, { brandGradient, renderStars, formatPrice } from '../components/ProductCard'

function StarInput({ value, onChange }) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          className={`bg-transparent border-none text-2xl p-0 leading-none transition ${n <= value ? 'text-accent' : 'text-border-light'}`}
          onClick={() => onChange(n)}
          aria-label={`Rate ${n} star${n > 1 ? 's' : ''}`}
        >
          {n <= value ? '★' : '☆'}
        </button>
      ))}
    </div>
  )
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  if (isNaN(d.getTime())) return dateStr
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

function ProductDetail() {
  const { id } = useParams()
  const [product, setProduct] = useState(null)
  const [similar, setSimilar] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [formTitle, setFormTitle] = useState('')
  const [formText, setFormText] = useState('')
  const [formRating, setFormRating] = useState(5)
  const [formAuthor, setFormAuthor] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [pendingReview, setPendingReview] = useState(null)
  const [overrideLabel, setOverrideLabel] = useState(null)
  const [showToast, setShowToast] = useState(false)
  const [reviewPage, setReviewPage] = useState(1)
  const REVIEWS_PER_PAGE = 5

  const newReviewRef = useRef(null)
  const reviewsSectionRef = useRef(null)

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      setReviewPage(1)
      try {
        const [prodRes, simRes] = await Promise.all([
          fetch(`/api/products/${id}`),
          fetch(`/api/products/${id}/similar`)
        ])
        if (!prodRes.ok) throw new Error('Failed to load product')
        const prodData = await prodRes.json()
        setProduct(prodData)
        if (simRes.ok) {
          const simData = await simRes.json()
          setSimilar(Array.isArray(simData) ? simData : simData.similar || simData.products || [])
        }
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [id])

  const handleSubmitReview = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    setPrediction(null)
    setPendingReview(null)
    setOverrideLabel(null)
    try {
      const res = await fetch('/api/reviews', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          product_id: Number(id),
          review_title: formTitle,
          review_text: formText,
          review_rating: Number(formRating),
          author: formAuthor
        })
      })
      if (!res.ok) throw new Error('Failed to submit review')
      const data = await res.json()
      setPrediction(data.prediction)
      setPendingReview(data.review)
      setOverrideLabel(data.prediction.label)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const handleConfirm = () => {
    const confirmed = {
      ...pendingReview,
      is_a_buyer: overrideLabel === 'Likely Buyer',
      _label: overrideLabel,
      _isNew: true
    }
    setProduct((prev) => ({
      ...prev,
      reviews: [confirmed, ...(prev.reviews || [])]
    }))
    setPrediction(null)
    setPendingReview(null)
    setOverrideLabel(null)
    setFormTitle('')
    setFormText('')
    setFormRating(5)
    setFormAuthor('')
    setReviewPage(1)

    setShowToast(true)
    setTimeout(() => setShowToast(false), 3000)
    setTimeout(() => {
      if (newReviewRef.current) {
        newReviewRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }, 100)
  }

  if (loading) return <div className="text-center py-16 px-6 text-text-light text-lg">Loading product...</div>
  if (error) return <div className="text-center py-16 px-6 text-danger text-lg">{error}</div>
  if (!product) return <div className="text-center py-16 px-6 text-danger text-lg">Product not found</div>

  const reviews = product.reviews || []

  const sortedReviews = [...reviews].sort((a, b) => {
    if (a._isNew && !b._isNew) return -1
    if (!a._isNew && b._isNew) return 1
    const dateA = a.review_date ? new Date(a.review_date) : new Date(0)
    const dateB = b.review_date ? new Date(b.review_date) : new Date(0)
    return dateB - dateA
  })

  const detailStars = product.avg_product_rating != null
    ? renderStars(product.avg_product_rating)
    : null

  return (
    <section className="max-w-[1200px] mx-auto px-6 py-8 pb-16">
      {showToast && <div className="fixed top-20 right-6 bg-success text-white px-6 py-3 rounded-lg shadow-lg z-[1000] font-semibold text-sm animate-[slideIn_0.3s_ease]">Review added successfully!</div>}

      <section className="grid grid-cols-1 md:grid-cols-2 gap-5 md:gap-10 mb-12">
        <div className="relative h-[220px] md:h-[360px] rounded-2xl overflow-hidden">
          {product.image && (
            <img
              src={product.image}
              alt={product.product_title}
              className="absolute inset-0 h-full w-full object-cover"
            />
          )}
          <div
            className="absolute inset-0 opacity-20"
            style={{ background: brandGradient(product.brand_name) }}
          />
        </div>
        <div className="flex flex-col justify-center">
          <h1 className="text-[22px] md:text-[28px] font-bold text-primary leading-snug mb-2">{product.product_title}</h1>
          <p className="text-base text-text-light mb-4">{product.brand_name}</p>
          {product.price != null && (
            <p className="text-[32px] font-bold text-secondary mb-4">${formatPrice(product.price)}</p>
          )}
          <div className="flex items-center gap-4">
            {detailStars && (
              <span className="text-xl font-semibold text-text flex items-center gap-1">
                <span className="text-[22px] text-accent">{detailStars.filled}<span className="text-border-light">{detailStars.empty}</span></span>
                <span>{Number(product.avg_product_rating).toFixed(1)}</span>
              </span>
            )}
            {product.review_count != null && (
              <span className="text-sm text-text-light">{product.review_count} reviews</span>
            )}
          </div>
        </div>
      </section>

      {similar.length > 0 && (
        <section className="mb-12">
          <h2 className="text-[22px] font-semibold text-primary mb-4">You May Also Like</h2>
          <div className="flex gap-4 overflow-x-auto pb-2">
            {similar.map((p) => (
              <ProductCard key={p.product_id} product={p} compact />
            ))}
          </div>
        </section>
      )}

      <section className="mb-12" ref={reviewsSectionRef}>
        <h2 className="text-[22px] font-semibold text-primary mb-4">Customer Reviews ({reviews.length})</h2>
        {(() => {
          const totalPages = Math.max(1, Math.ceil(sortedReviews.length / REVIEWS_PER_PAGE))
          const startIdx = (reviewPage - 1) * REVIEWS_PER_PAGE
          const pageReviews = sortedReviews.slice(startIdx, startIdx + REVIEWS_PER_PAGE)

          const goToPage = (page) => {
            setReviewPage(page)
            if (reviewsSectionRef.current) {
              reviewsSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
            }
          }

          return (
            <>
              <div className="flex flex-col gap-4">
                {pageReviews.map((review, idx) => {
                  const stars = renderStars(review.review_rating)
                  const globalIdx = startIdx + idx
                  const isFirst = globalIdx === 0 && review._isNew
                  return (
                    <article
                      key={review.review_id || globalIdx}
                      className="bg-surface rounded-xl p-5 border-l-4 border-secondary shadow-sm"
                      ref={isFirst ? newReviewRef : undefined}
                    >
                      <div className="flex items-center gap-3 mb-2 flex-wrap">
                        <span className="text-sm font-semibold text-text flex items-center gap-0.5">
                          <span className="text-[22px] text-accent">{stars.filled}<span className="text-border-light">{stars.empty}</span></span>
                          <span>{Number(review.review_rating).toFixed(1)}</span>
                        </span>
                        <strong className="text-[15px] text-primary font-semibold">{review.review_title}</strong>
                        {review.is_a_buyer ? (
                          <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-success/20 text-success">Verified Buyer</span>
                        ) : (
                          <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-bg-light text-text-light">Not a Buyer</span>
                        )}
                      </div>
                      <p className="text-sm text-text-light leading-relaxed mb-2">{review.review_text}</p>
                      <div className="flex items-center gap-4 text-xs text-text-lighter">
                        {review.author && <span>{review.author}</span>}
                        {review.review_date && (
                          <span>{formatDate(review.review_date)}</span>
                        )}
                      </div>
                    </article>
                  )
                })}
                {reviews.length === 0 && (
                  <p className="text-text-light text-[15px]">No reviews yet. Be the first to review this product.</p>
                )}
              </div>
              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-3 pt-5 pb-2">
                  <button
                    className="h-9 px-4 border border-border rounded-lg bg-surface text-primary text-[13px] font-medium hover:bg-secondary hover:text-white hover:border-secondary disabled:opacity-40 disabled:cursor-default transition"
                    disabled={reviewPage === 1}
                    onClick={() => goToPage(reviewPage - 1)}
                  >
                    Previous
                  </button>
                  <div className="flex gap-1 items-center">
                    {Array.from({ length: totalPages }, (_, i) => i + 1)
                      .filter(p => p === 1 || p === totalPages || Math.abs(p - reviewPage) <= 1)
                      .reduce((acc, p, i, arr) => {
                        if (i > 0 && p - arr[i - 1] > 1) acc.push('...')
                        acc.push(p)
                        return acc
                      }, [])
                      .map((p, i) =>
                        p === '...' ? (
                          <span key={`dots-${i}`} className="w-[34px] text-center text-text-light text-[13px]">...</span>
                        ) : (
                          <button
                            key={p}
                            className={`w-[34px] h-[34px] border rounded-lg flex items-center justify-center text-[13px] transition ${p === reviewPage ? 'bg-secondary text-white border-secondary' : 'border-border bg-surface text-primary hover:border-secondary hover:text-secondary'}`}
                            onClick={() => goToPage(p)}
                          >
                            {p}
                          </button>
                        )
                      )}
                  </div>
                  <button
                    className="h-9 px-4 border border-border rounded-lg bg-surface text-primary text-[13px] font-medium hover:bg-secondary hover:text-white hover:border-secondary disabled:opacity-40 disabled:cursor-default transition"
                    disabled={reviewPage === totalPages}
                    onClick={() => goToPage(reviewPage + 1)}
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )
        })()}
      </section>

      <section className="mb-12">
        <h2 className="text-[22px] font-semibold text-primary mb-4">Write a Review</h2>
        <form className="flex flex-col gap-4 max-w-[600px]" onSubmit={handleSubmitReview}>
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-semibold text-primary">Title</span>
            <input
              type="text"
              className="px-3.5 py-2.5 border border-border rounded-lg text-sm text-text bg-surface outline-none focus:border-secondary transition"
              value={formTitle}
              onChange={(e) => setFormTitle(e.target.value)}
              required
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-semibold text-primary">Review</span>
            <textarea
              rows={4}
              className="px-3.5 py-2.5 border border-border rounded-lg text-sm text-text bg-surface outline-none focus:border-secondary transition resize-y"
              value={formText}
              onChange={(e) => setFormText(e.target.value)}
              required
            />
          </label>
          <div className="flex flex-col gap-1.5">
            <span className="text-sm font-semibold text-primary">Rating</span>
            <StarInput value={formRating} onChange={setFormRating} />
          </div>
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-semibold text-primary">Your Name</span>
            <input
              type="text"
              className="px-3.5 py-2.5 border border-border rounded-lg text-sm text-text bg-surface outline-none focus:border-secondary transition"
              value={formAuthor}
              onChange={(e) => setFormAuthor(e.target.value)}
              required
            />
          </label>
          <button type="submit" className="self-start h-11 px-7 border-none rounded-lg bg-secondary text-white text-[15px] font-semibold hover:bg-secondary-light transition disabled:opacity-50 disabled:cursor-default" disabled={submitting}>
            {submitting ? 'Submitting...' : 'Submit Review'}
          </button>
        </form>

        {prediction && (
          <div className="mt-8 max-w-[600px] bg-surface rounded-xl p-6 shadow-md">
            <div>
              <h3 className="text-lg mb-4 text-primary font-semibold">AI Prediction</h3>
              <div className="flex items-center gap-4 mb-3 max-md:flex-col max-md:items-start max-md:gap-2">
                <div className="text-[42px] font-bold text-primary leading-none">
                  {(prediction.probability * 100).toFixed(0)}%
                </div>
                <div className={`inline-block px-4 py-1.5 rounded-md text-base font-bold ${prediction.label === 'Likely Buyer' ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'}`}>
                  {prediction.label}
                </div>
              </div>
              <div>
                <div className="flex-1 h-2.5 bg-bg-light rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-[width] duration-400 ${prediction.label === 'Likely Buyer' ? 'bg-success' : 'bg-danger'}`}
                    style={{ width: `${(prediction.probability * 100).toFixed(0)}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-3 mb-6 mt-6">
              {Object.entries(prediction.models).map(([key, model]) => (
                <div key={key} className="flex items-center gap-3 px-3.5 py-2.5 bg-bg-light rounded-lg">
                  <span className="text-[13px] font-semibold text-text-light min-w-40 max-md:min-w-[100px]">{model.name}</span>
                  <div className="flex-1 h-2.5 bg-bg-light rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-[width] duration-400 ${model.probability >= 0.5 ? 'bg-success' : 'bg-danger'}`}
                      style={{ width: `${(model.probability * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="text-[13px] font-semibold text-text whitespace-nowrap min-w-12 text-right">{(model.probability * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <div className="flex items-center gap-2.5 mb-4 flex-wrap max-md:flex-col max-md:items-start">
              <span className="text-sm font-semibold text-text-light">Override label:</span>
              <button
                type="button"
                className={`h-[34px] px-3.5 border-2 rounded-md text-[13px] font-semibold transition ${overrideLabel === 'Likely Buyer' ? 'border-success bg-success/20 text-success' : 'border-border bg-surface text-text-light hover:border-text-lighter'}`}
                onClick={() => setOverrideLabel('Likely Buyer')}
              >
                Likely Buyer
              </button>
              <button
                type="button"
                className={`h-[34px] px-3.5 border-2 rounded-md text-[13px] font-semibold transition ${overrideLabel === 'Unlikely Buyer' ? 'border-danger bg-danger/20 text-danger' : 'border-border bg-surface text-text-light hover:border-text-lighter'}`}
                onClick={() => setOverrideLabel('Unlikely Buyer')}
              >
                Unlikely Buyer
              </button>
            </div>

            <button type="button" className="h-[42px] px-8 border-none rounded-lg bg-primary text-white text-[15px] font-semibold hover:bg-primary-light transition" onClick={handleConfirm}>
              Confirm
            </button>
          </div>
        )}
      </section>
    </section>
  )
}

export default ProductDetail
