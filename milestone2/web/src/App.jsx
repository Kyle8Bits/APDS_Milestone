import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import './App.css'

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

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main className="pt-[60px] min-h-screen bg-bg">
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/search" element={<Search />} />
            <Route path="/products/:id" element={<ProductDetail />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </Suspense>
      </main>
    </BrowserRouter>
  )
}

export default App
