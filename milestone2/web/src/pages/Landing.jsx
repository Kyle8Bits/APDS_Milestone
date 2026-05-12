import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import ProductCard from '../components/ProductCard'

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i = 0) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.6, ease: 'easeOut' } })
}

const stagger = {
  visible: { transition: { staggerChildren: 0.08 } }
}

function Landing() {
  const [topRated, setTopRated] = useState([])
  const [brands, setBrands] = useState([])
  const [stats, setStats] = useState(null)
  const carouselRef = useRef(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/top-rated?n=12').then(r => r.ok ? r.json() : []),
      fetch('/api/brands').then(r => r.ok ? r.json() : []),
      fetch('/api/stats').then(r => r.ok ? r.json() : null),
    ]).then(([top, br, st]) => {
      setTopRated(Array.isArray(top) ? top : [])
      setBrands(Array.isArray(br) ? br : [])
      setStats(st)
    })
  }, [])

  const scrollCarousel = (dir) => {
    if (!carouselRef.current) return
    carouselRef.current.scrollBy({ left: dir * 340, behavior: 'smooth' })
  }

  return (
    <div className="overflow-hidden">
      {/* Hero */}
      <section className="relative min-h-[85vh] flex items-center justify-center overflow-hidden bg-gradient-to-br from-primary via-primary-light to-primary">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-[10%] left-[5%] w-72 h-72 rounded-full bg-secondary blur-[100px]" />
          <div className="absolute bottom-[15%] right-[10%] w-96 h-96 rounded-full bg-accent blur-[120px]" />
          <div className="absolute top-[50%] left-[50%] w-64 h-64 rounded-full bg-secondary-light blur-[80px]" />
        </div>

        <div className="relative z-10 max-w-[900px] mx-auto px-6 text-center">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={stagger}
          >
            <motion.p
              variants={fadeUp}
              custom={0}
              className="text-secondary-lighter text-sm font-semibold uppercase tracking-[0.2em] mb-4"
            >
              Your Beauty Destination
            </motion.p>
            <motion.h1
              variants={fadeUp}
              custom={1}
              className="text-[56px] max-md:text-[36px] font-bold text-white leading-[1.1] mb-6 tracking-tight"
            >
              Discover Your
              <span className="block text-accent">Perfect Glow</span>
            </motion.h1>
            <motion.p
              variants={fadeUp}
              custom={2}
              className="text-lg max-md:text-base text-white/70 max-w-[560px] mx-auto mb-10 leading-relaxed"
            >
              Browse hundreds of top-rated cosmetics and beauty products. Read authentic reviews, find your perfect match, and shop with confidence.
            </motion.p>
            <motion.div
              variants={fadeUp}
              custom={3}
              className="flex items-center justify-center gap-4 max-md:flex-col"
            >
              <Link
                to="/products"
                className="inline-flex items-center gap-2 h-14 px-10 rounded-xl bg-secondary text-white text-[17px] font-semibold hover:bg-secondary-light transition-all hover:scale-105 shadow-lg shadow-secondary/30"
              >
                Shop Now
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
              </Link>
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 h-14 px-10 rounded-xl bg-white/10 text-white text-[17px] font-semibold hover:bg-white/20 transition-all backdrop-blur-sm border border-white/20"
              >
                View Analytics
              </Link>
            </motion.div>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 1 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <div className="w-6 h-10 rounded-full border-2 border-white/30 flex items-start justify-center p-1.5">
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
              className="w-1.5 h-1.5 rounded-full bg-white/60"
            />
          </div>
        </motion.div>
      </section>

      {/* Stats Bar */}
      {stats && (
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="bg-surface border-y border-border-light"
        >
          <div className="max-w-[1200px] mx-auto px-6 py-8 grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="text-[28px] font-bold text-primary">{stats.total_products?.toLocaleString()}</div>
              <div className="text-sm text-text-light">Products</div>
            </div>
            <div>
              <div className="text-[28px] font-bold text-primary">{stats.total_reviews?.toLocaleString()}</div>
              <div className="text-sm text-text-light">Reviews</div>
            </div>
            <div>
              <div className="text-[28px] font-bold text-primary">{stats.total_brands}</div>
              <div className="text-sm text-text-light">Brands</div>
            </div>
            <div>
              <div className="text-[28px] font-bold text-accent">
                {stats.sentiment ? `${(stats.sentiment.avg_compound * 100).toFixed(0)}%` : '--'}
              </div>
              <div className="text-sm text-text-light">Positive Sentiment</div>
            </div>
          </div>
        </motion.section>
      )}

      {/* Top Rated Carousel */}
      {topRated.length > 0 && (
        <section className="max-w-[1200px] mx-auto px-6 py-16">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-80px' }}
            variants={stagger}
          >
            <motion.div variants={fadeUp} className="flex items-end justify-between mb-8">
              <div>
                <p className="text-secondary text-sm font-semibold uppercase tracking-[0.15em] mb-1">Curated Selection</p>
                <h2 className="text-[28px] font-bold text-primary">Top Rated Products</h2>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => scrollCarousel(-1)}
                  className="w-10 h-10 flex items-center justify-center rounded-full border border-border bg-surface text-primary text-lg hover:bg-secondary hover:text-white hover:border-secondary transition"
                >
                  &#8249;
                </button>
                <button
                  onClick={() => scrollCarousel(1)}
                  className="w-10 h-10 flex items-center justify-center rounded-full border border-border bg-surface text-primary text-lg hover:bg-secondary hover:text-white hover:border-secondary transition"
                >
                  &#8250;
                </button>
              </div>
            </motion.div>
            <motion.div variants={fadeUp}>
              <div ref={carouselRef} className="flex gap-5 overflow-x-auto pb-4 scrollbar-thin scroll-smooth">
                {topRated.map((p, i) => (
                  <motion.div
                    key={p.product_id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.05, duration: 0.4 }}
                  >
                    <ProductCard product={p} compact />
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        </section>
      )}

      {/* Brands */}
      {brands.length > 0 && (
        <section className="bg-surface border-y border-border-light">
          <div className="max-w-[1200px] mx-auto px-6 py-16">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: '-80px' }}
              variants={stagger}
            >
              <motion.div variants={fadeUp} className="text-center mb-10">
                <p className="text-secondary text-sm font-semibold uppercase tracking-[0.15em] mb-1">Our Partners</p>
                <h2 className="text-[28px] font-bold text-primary">Shop by Brand</h2>
              </motion.div>
              <motion.div
                variants={stagger}
                className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4"
              >
                {brands.map((brand) => (
                  <motion.div key={brand} variants={fadeUp}>
                    <Link
                      to={`/products?brand=${encodeURIComponent(brand)}`}
                      className="block text-center py-5 px-4 rounded-xl border border-border-light hover:border-secondary hover:shadow-md transition-all bg-bg hover:bg-bg-light group"
                    >
                      <span className="text-[15px] font-semibold text-primary group-hover:text-secondary transition-colors">{brand}</span>
                    </Link>
                  </motion.div>
                ))}
              </motion.div>
            </motion.div>
          </div>
        </section>
      )}

      {/* Features / Why Us */}
      <section className="max-w-[1200px] mx-auto px-6 py-16">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-80px' }}
          variants={stagger}
        >
          <motion.div variants={fadeUp} className="text-center mb-12">
            <p className="text-secondary text-sm font-semibold uppercase tracking-[0.15em] mb-1">Why GlowCart</p>
            <h2 className="text-[28px] font-bold text-primary">Smart Beauty Shopping</h2>
          </motion.div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: (
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                ),
                title: 'Authentic Reviews',
                desc: 'Real reviews from verified buyers with AI-powered sentiment analysis to help you decide.'
              },
              {
                icon: (
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
                ),
                title: 'Smart Search',
                desc: 'Fuzzy search that understands what you mean — even with typos or partial brand names.'
              },
              {
                icon: (
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                ),
                title: 'ML Recommendations',
                desc: 'Machine learning models analyze text and metadata to predict buyer intent and recommend products.'
              },
            ].map((feature, i) => (
              <motion.div
                key={feature.title}
                variants={fadeUp}
                custom={i}
                className="text-center p-8 rounded-2xl bg-surface border border-border-light hover:shadow-lg hover:-translate-y-1 transition-all"
              >
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-secondary/10 text-secondary mb-5">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-bold text-primary mb-3">{feature.title}</h3>
                <p className="text-sm text-text-light leading-relaxed">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* CTA */}
      <section className="bg-gradient-to-br from-primary to-primary-light">
        <div className="max-w-[800px] mx-auto px-6 py-20 text-center">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
          >
            <motion.h2 variants={fadeUp} className="text-[32px] max-md:text-[24px] font-bold text-white mb-4">
              Ready to Find Your Next Favorite?
            </motion.h2>
            <motion.p variants={fadeUp} custom={1} className="text-white/70 text-lg mb-8 max-w-[480px] mx-auto">
              Explore our curated collection of beauty products and discover what thousands of reviewers love.
            </motion.p>
            <motion.div variants={fadeUp} custom={2}>
              <Link
                to="/products"
                className="inline-flex items-center gap-2 h-14 px-12 rounded-xl bg-secondary text-white text-[17px] font-semibold hover:bg-secondary-light transition-all hover:scale-105 shadow-lg shadow-black/20"
              >
                Browse All Products
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-primary-fade border-t border-border-light">
        <div className="max-w-[1200px] mx-auto px-6 py-8 flex items-center justify-between max-md:flex-col max-md:gap-3 text-sm text-text-light">
          <span className="font-semibold text-primary">GlowCart</span>
          <span>COSC3801/3015 Advanced Programming for Data Science</span>
        </div>
      </footer>
    </div>
  )
}

export default Landing
