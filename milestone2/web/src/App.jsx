import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Navbar from './components/Navbar'
import './App.css'

const Landing = lazy(() => import('./pages/Landing'))
const Home = lazy(() => import('./pages/Home'))
const Search = lazy(() => import('./pages/Search'))
const ProductDetail = lazy(() => import('./pages/ProductDetail'))
const Dashboard = lazy(() => import('./pages/Dashboard'))

function PageLoader() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-text-light">
      <div className="w-8 h-8 border-3 border-border-light border-t-secondary rounded-full animate-spin mb-3" />
      <span className="text-base">Loading...</span>
    </div>
  )
}

const pageTransition = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.2, ease: 'easeIn' } },
}

function AnimatedRoutes() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <motion.div key={location.pathname} {...pageTransition}>
        <Suspense fallback={<PageLoader />}>
          <Routes location={location}>
            <Route path="/" element={<Landing />} />
            <Route path="/products" element={<Home />} />
            <Route path="/search" element={<Search />} />
            <Route path="/products/:id" element={<ProductDetail />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </Suspense>
      </motion.div>
    </AnimatePresence>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main className="pt-[60px] min-h-screen bg-bg">
        <AnimatedRoutes />
      </main>
    </BrowserRouter>
  )
}

export default App
