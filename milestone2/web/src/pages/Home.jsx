import { useState, useEffect, useMemo, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import Slider from 'rc-slider'
import 'rc-slider/assets/index.css'
import ProductCard from '../components/ProductCard'

const PRICE_MIN = 0
const PRICE_MAX = 3000

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } }
}

function Home() {
  const [searchParams] = useSearchParams()
  const initialBrand = searchParams.get('brand') || ''

  const [products, setProducts] = useState([])
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [brands, setBrands] = useState([])
  const [selectedBrand, setSelectedBrand] = useState(initialBrand)
  const [sortBy, setSortBy] = useState('name')
  const [priceRange, setPriceRange] = useState([PRICE_MIN, PRICE_MAX])
  const [committedPrice, setCommittedPrice] = useState([PRICE_MIN, PRICE_MAX])
  const [minRating, setMinRating] = useState('')

  useEffect(() => {
    fetch('/api/brands')
      .then((res) => res.ok ? res.json() : Promise.reject('Failed to load brands'))
      .then((data) => setBrands(Array.isArray(data) ? data : data.brands || []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    const fetchProducts = async () => {
      setLoading(true)
      setError(null)
      try {
        const params = new URLSearchParams({ page, per_page: 20, sort_by: sortBy })
        if (selectedBrand) params.set('brand', selectedBrand)
        if (committedPrice[0] > PRICE_MIN) params.set('min_price', committedPrice[0])
        if (committedPrice[1] < PRICE_MAX) params.set('max_price', committedPrice[1])
        if (minRating) params.set('min_rating', minRating)
        const res = await fetch(`/api/products?${params}`)
        if (!res.ok) throw new Error('Failed to fetch products')
        const data = await res.json()
        setProducts(data.products || [])
        setTotalPages(data.pages || 1)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchProducts()
  }, [page, selectedBrand, sortBy, committedPrice, minRating])

  const handleBrandChange = (e) => {
    setSelectedBrand(e.target.value)
    setPage(1)
  }

  const handleSortChange = (e) => {
    setSortBy(e.target.value)
    setPage(1)
  }

  const handlePriceSlide = useCallback((val) => {
    setPriceRange(val)
  }, [])

  const handlePriceCommit = useCallback((val) => {
    setCommittedPrice(val)
    setPage(1)
  }, [])

  const handleRatingChange = (e) => {
    setMinRating(e.target.value)
    setPage(1)
  }

  const handleClearFilters = () => {
    setSelectedBrand('')
    setSortBy('name')
    setPriceRange([PRICE_MIN, PRICE_MAX])
    setCommittedPrice([PRICE_MIN, PRICE_MAX])
    setMinRating('')
    setPage(1)
  }

  const hasFilters = selectedBrand || minRating
    || committedPrice[0] > PRICE_MIN || committedPrice[1] < PRICE_MAX

  const pageNumbers = useMemo(() => {
    const pages = []
    let start = Math.max(1, page - 2)
    let end = Math.min(totalPages, page + 2)
    if (page <= 2) end = Math.min(totalPages, 5)
    if (page >= totalPages - 1) start = Math.max(1, totalPages - 4)
    for (let i = start; i <= end; i++) pages.push(i)
    return pages
  }, [page, totalPages])

  return (
    <section>
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="text-center py-16 px-6 pt-16 pb-12 bg-gradient-to-br from-primary to-primary-light text-white"
      >
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
          className="text-[42px] max-md:text-[28px] font-bold text-white mb-3 tracking-tight"
        >
          {selectedBrand || 'All Products'}
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="text-lg text-white/75 max-w-[520px] mx-auto"
        >
          Browse, filter, and find your perfect beauty match.
        </motion.p>
      </motion.section>

      <motion.div
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        className="max-w-[1200px] mx-auto px-6 pt-6"
      >
        <div className="flex gap-3 items-center flex-wrap max-md:flex-col max-md:items-stretch">
          <select className="px-3 py-2 border border-border rounded-lg bg-surface text-text text-[0.9rem] outline-none focus:border-secondary" value={selectedBrand} onChange={handleBrandChange}>
            <option value="">All Brands</option>
            {brands.map((brand) => (
              <option key={brand} value={brand}>{brand}</option>
            ))}
          </select>
          <select className="px-3 py-2 border border-border rounded-lg bg-surface text-text text-[0.9rem] outline-none focus:border-secondary" value={sortBy} onChange={handleSortChange}>
            <option value="name">Name (A-Z)</option>
            <option value="rating">Rating (High to Low)</option>
            <option value="price_asc">Price (Low to High)</option>
            <option value="price_desc">Price (High to Low)</option>
            <option value="reviews">Most Reviews</option>
          </select>
          <div className="flex flex-col gap-1.5 min-w-[200px] py-1">
            <span className="text-[0.82rem] text-text-light font-medium whitespace-nowrap">
              Price: ${priceRange[0]} — ${priceRange[1]}
            </span>
            <Slider
              range
              min={PRICE_MIN}
              max={PRICE_MAX}
              step={50}
              value={priceRange}
              onChange={handlePriceSlide}
              onChangeComplete={handlePriceCommit}
              styles={{
                track: { backgroundColor: 'var(--color-secondary)', height: 6 },
                handle: {
                  borderColor: 'var(--color-secondary)',
                  backgroundColor: 'var(--color-surface)',
                  width: 18,
                  height: 18,
                  marginTop: -6,
                  boxShadow: '0 2px 4px rgba(0,0,0,0.15)',
                  opacity: 1,
                },
                rail: { backgroundColor: 'var(--color-border)', height: 6 },
              }}
            />
          </div>
          <select className="px-3 py-2 border border-border rounded-lg bg-surface text-text text-[0.9rem] outline-none focus:border-secondary" value={minRating} onChange={handleRatingChange}>
            <option value="">All Ratings</option>
            <option value="4">4+ Stars</option>
            <option value="3.5">3.5+ Stars</option>
            <option value="3">3+ Stars</option>
            <option value="2">2+ Stars</option>
          </select>
          {hasFilters && (
            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="px-3.5 py-2 border border-secondary rounded-lg bg-transparent text-secondary text-[0.85rem] font-medium whitespace-nowrap hover:bg-secondary hover:text-white transition"
              onClick={handleClearFilters}
            >
              Clear Filters
            </motion.button>
          )}
        </div>
      </motion.div>

      {loading && <div className="text-center py-16 px-6 text-text-light text-lg">Loading products...</div>}
      {error && <div className="text-center py-16 px-6 text-danger text-lg">{error}</div>}

      {!loading && !error && (
        <>
          <section className="max-w-[1200px] mx-auto px-6 py-8 grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-6 max-md:grid-cols-1 max-md:px-4 max-md:py-5">
            {products.map((product, i) => (
              <motion.div
                key={product.product_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.03, 0.4), duration: 0.4 }}
              >
                <ProductCard product={product} />
              </motion.div>
            ))}
          </section>

          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-4 px-6 py-4 pb-12 max-md:gap-2 max-md:flex-wrap">
              <button
                className="h-10 px-5 border border-border rounded-lg bg-surface text-primary text-sm font-medium hover:bg-secondary hover:text-white hover:border-secondary disabled:opacity-40 disabled:cursor-default transition"
                disabled={page <= 1}
                onClick={() => setPage(page - 1)}
              >
                Previous
              </button>
              <div className="flex gap-1 items-center">
                {pageNumbers[0] > 1 && (
                  <>
                    <button className="w-9 h-9 border border-border rounded-lg bg-surface flex items-center justify-center text-[0.9rem] text-primary hover:border-secondary hover:text-secondary transition" onClick={() => setPage(1)}>1</button>
                    {pageNumbers[0] > 2 && <span className="w-9 h-9 flex items-center justify-center text-[0.9rem] text-text-light">...</span>}
                  </>
                )}
                {pageNumbers.map((num) => (
                  <button
                    key={num}
                    className={`w-9 h-9 border rounded-lg flex items-center justify-center text-[0.9rem] transition ${num === page ? 'bg-secondary text-white border-secondary' : 'border-border bg-surface text-primary hover:border-secondary hover:text-secondary'}`}
                    onClick={() => setPage(num)}
                  >
                    {num}
                  </button>
                ))}
                {pageNumbers[pageNumbers.length - 1] < totalPages && (
                  <>
                    {pageNumbers[pageNumbers.length - 1] < totalPages - 1 && (
                      <span className="w-9 h-9 flex items-center justify-center text-[0.9rem] text-text-light">...</span>
                    )}
                    <button className="w-9 h-9 border border-border rounded-lg bg-surface flex items-center justify-center text-[0.9rem] text-primary hover:border-secondary hover:text-secondary transition" onClick={() => setPage(totalPages)}>
                      {totalPages}
                    </button>
                  </>
                )}
              </div>
              <button
                className="h-10 px-5 border border-border rounded-lg bg-surface text-primary text-sm font-medium hover:bg-secondary hover:text-white hover:border-secondary disabled:opacity-40 disabled:cursor-default transition"
                disabled={page >= totalPages}
                onClick={() => setPage(page + 1)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </section>
  )
}

export default Home
