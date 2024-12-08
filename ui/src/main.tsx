import { ThemeProvider, createTheme } from '@mui/material/styles';
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import "./index.css"
import App from './App.tsx'

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1a1a1a',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#242424', // Example background color
    },
    text: {
      primary: '#ffffff', // Example text color
    },
  },
});
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
    <App />
    </ThemeProvider>
  </StrictMode>,
)
