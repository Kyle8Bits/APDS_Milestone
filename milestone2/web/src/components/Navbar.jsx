import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'

function Navbar() {
  const [query, setQuery] = useState('')
  const [dark, setDark] = useState(() => localStorage.getItem('theme') === 'dark')
  const navigate = useNavigate()

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = query.trim()
    if (trimmed) {
      navigate(`/search?q=${encodeURIComponent(trimmed)}`)
    }
  }

  return (
    <nav className="fixed top-0 w-full h-[60px] bg-primary dark:bg-[#1a1a1a] z-50 shadow">
      <div className="max-w-[1200px] mx-auto h-full flex items-center px-6 gap-6">
        <Link to="/" className="text-[22px] font-bold text-white hover:text-secondary transition-colors">GlowCart</Link>
        <form className="flex-1 max-w-[480px] flex" onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Search products..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 h-[38px] px-3.5 border-none rounded-l-lg bg-white/12 text-white text-sm placeholder:text-white/50 focus:bg-white/20 focus:outline-none transition-colors"
          />
          <button type="submit" className="h-[38px] px-[18px] border-none rounded-r-lg bg-secondary text-white text-sm font-semibold hover:bg-secondary-light transition-colors cursor-pointer">Search</button>
        </form>
        <div className="ml-auto flex items-center gap-3">
          <Link to="/dashboard" className="text-sm font-semibold text-white/85 px-3.5 py-1.5 rounded-lg hover:text-white hover:bg-secondary/20 transition-colors">Dashboard</Link>
          <button
            onClick={() => setDark(!dark)}
            className="w-9 h-9 flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/20 text-white text-lg transition-colors"
            aria-label="Toggle dark mode"
          >
            {dark ? '☀️' : '🌙'}
          </button>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
