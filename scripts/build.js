/**
 * Production Build Script for Vercel Deployment
 * AI-Powered Satellite Wildfire & Burned-Area Intelligence Platform
 */

const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const appDir = path.join(projectRoot, 'app');
const publicDir = path.join(projectRoot, 'public');

console.log('--- BUILDING PRODUCTION ASSETS FOR VERCEL ---');

// 1. Ensure public directories exist
fs.mkdirSync(publicDir, { recursive: true });
fs.mkdirSync(path.join(publicDir, 'static', 'css'), { recursive: true });
fs.mkdirSync(path.join(publicDir, 'static', 'js'), { recursive: true });

// 2. Process and copy index.html
const srcHtmlPath = path.join(appDir, 'templates', 'index.html');
const destHtmlPath = path.join(publicDir, 'index.html');

if (fs.existsSync(srcHtmlPath)) {
    let htmlContent = fs.readFileSync(srcHtmlPath, 'utf8');
    
    // Ensure Title & Branding conform to Production Title
    htmlContent = htmlContent.replace(
        /<title>.*?<\/title>/i,
        '<title>AI-Powered Satellite Wildfire & Burned-Area Intelligence Platform</title>'
    );
    
    fs.writeFileSync(destHtmlPath, htmlContent, 'utf8');
    console.log('✓ Copied & prepared: public/index.html');
} else {
    console.error('Error: app/templates/index.html not found!');
    process.exit(1);
}

// 3. Copy CSS
const srcCssPath = path.join(appDir, 'static', 'css', 'style.css');
const destCssPath = path.join(publicDir, 'static', 'css', 'style.css');
if (fs.existsSync(srcCssPath)) {
    fs.copyFileSync(srcCssPath, destCssPath);
    console.log('✓ Copied: public/static/css/style.css');
}

// 4. Copy JavaScript
const srcJsPath = path.join(appDir, 'static', 'js', 'app.js');
const destJsPath = path.join(publicDir, 'static', 'js', 'app.js');
if (fs.existsSync(srcJsPath)) {
    fs.copyFileSync(srcJsPath, destJsPath);
    console.log('✓ Copied: public/static/js/app.js');
}

console.log('--- PRODUCTION BUILD COMPLETE ---');
