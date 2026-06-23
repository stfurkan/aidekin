import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { SiteApp } from '@/site/SiteApp'
import { installErrorCapture } from '@/core/diagnostics'
import '@/index.css'

installErrorCapture()

const root = document.getElementById('root')
if (!root) throw new Error('Aidekin: #root mount point missing from index.html')

createRoot(root).render(
  <StrictMode>
    <BrowserRouter>
      <SiteApp />
    </BrowserRouter>
  </StrictMode>,
)
