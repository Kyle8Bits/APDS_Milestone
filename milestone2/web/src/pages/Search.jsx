import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import ProductCard from '../components/ProductCard'

function Search() {
  const [searchParams] = useSearchParams()
  const query = searchParams.get('q') || ''
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!query) {
      setResults([])
      return
    }

    const fetchResults = async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`)
        if (!res.ok) throw new Error('Search failed')
        const data = await res.json()
        setResults(data.products || data.results || [])
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchResults()
  }, [query])

  return (
    <section className="max-w-[1200px] mx-auto px-6 pt-8">
      {loading && (
        <div className="text-center py-16 px-6 text-text-light text-lg flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-border-light border-t-secondary rounded-full animate-spin" />
          Searching for &ldquo;{query}&rdquo;...
        </div>
      )}

      {error && <div className="text-center py-16 px-6 text-danger text-lg">{error}</div>}

      {!loading && !error && results.length > 0 && (
        <>
          <motion.h2
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="text-[22px] text-primary mb-2"
          >
            {results.length} products matched for &ldquo;{query}&rdquo;
          </motion.h2>
          <section className="max-w-[1200px] mx-auto px-6 py-8 grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-6">
            {results.map((product, i) => (
              <motion.div
                key={product.product_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.04, 0.5), duration: 0.4 }}
              >
                <ProductCard product={product} />
              </motion.div>
            ))}
          </section>
        </>
      )}

      {!loading && !error && results.length === 0 && query && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12 px-6 text-text-light"
        >
          <h3 className="text-xl mb-2 text-text">No products found for &ldquo;{query}&rdquo;</h3>
          <p className="text-[0.95rem] text-text-light">Try a different keyword or check the spelling.</p>
        </motion.div>
      )}

      {!query && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12 px-6 text-text-light"
        >
          <h3 className="text-xl mb-2 text-text">Search for beauty products</h3>
          <p className="text-[0.95rem] text-text-light">Enter a keyword above to find cosmetics and beauty products.</p>
        </motion.div>
      )}
    </section>
  )
}

export default Search
