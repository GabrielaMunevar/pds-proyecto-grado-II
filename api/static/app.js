// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE_URL = window.location.origin; // Auto-detect API URL
const API_ENDPOINTS = {
    classify: `${API_BASE_URL}/classify`,
    generate: `${API_BASE_URL}/generate`,
    health: `${API_BASE_URL}/health`
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

let currentTab = 'classify';

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeClassify();
    initializeGenerate();
    checkAPIHealth();
});

// ============================================================================
// TAB MANAGEMENT
// ============================================================================

function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    currentTab = tabName;

    // Update buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        }
    });

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Clear results when switching tabs
    clearResults();
}

// ============================================================================
// CLASSIFY FUNCTIONALITY
// ============================================================================

function initializeClassify() {
    const textarea = document.getElementById('classify-text');
    const charCount = document.getElementById('classify-char-count');
    const classifyBtn = document.getElementById('classify-btn');

    // Character counter
    textarea.addEventListener('input', () => {
        const count = textarea.value.length;
        charCount.textContent = count;
    });

    // Classify button
    classifyBtn.addEventListener('click', async () => {
        const text = textarea.value.trim();
        
        if (text.length < 10) {
            showToast('Please enter at least 10 characters', 'error');
            return;
        }

        await classifyText(text);
    });
}

async function classifyText(text) {
    showLoading(true);
    hideResult('classify-result');

    try {
        const response = await fetch(API_ENDPOINTS.classify, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Classification failed');
        }

        const result = await response.json();
        displayClassificationResult(result);
        showToast('Text classified successfully!', 'success');

    } catch (error) {
        console.error('Classification error:', error);
        showToast(error.message || 'Failed to classify text', 'error');
    } finally {
        showLoading(false);
    }
}

function displayClassificationResult(result) {
    const resultContainer = document.getElementById('classify-result');
    const badge = document.getElementById('classification-badge');
    const badgeIcon = badge.querySelector('.badge-icon');
    const badgeText = badge.querySelector('.badge-text');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceProgress = document.getElementById('confidence-progress');
    const freValue = document.getElementById('fre-value');
    const fkgValue = document.getElementById('fkg-value');
    const awlValue = document.getElementById('awl-value');
    const techValue = document.getElementById('tech-value');
    const reasoningText = document.getElementById('reasoning-text');

    // Update badge
    if (result.is_pls) {
        badge.className = 'classification-badge pls';
        badgeIcon.textContent = '';
        badgeText.textContent = 'Plain Language Summary';
    } else {
        badge.className = 'classification-badge non-pls';
        badgeIcon.textContent = '';
        badgeText.textContent = 'Technical Text';
    }

    // Update confidence
    const confidencePercent = Math.round(result.confidence * 100);
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceProgress.style.width = `${confidencePercent}%`;

    // Update metrics
    freValue.textContent = result.flesch_reading_ease.toFixed(1);
    fkgValue.textContent = result.flesch_kincaid_grade.toFixed(1);
    awlValue.textContent = result.avg_word_length.toFixed(2);
    techValue.textContent = result.technical_terms_count;

    // Update reasoning
    reasoningText.textContent = result.reasoning;

    // Show result
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// GENERATE FUNCTIONALITY
// ============================================================================

function initializeGenerate() {
    const textarea = document.getElementById('generate-text');
    const charCount = document.getElementById('generate-char-count');
    const generateBtn = document.getElementById('generate-btn');
    const copyBtn = document.getElementById('copy-pls-btn');

    // Character counter
    textarea.addEventListener('input', () => {
        const count = textarea.value.length;
        charCount.textContent = count;
    });

    // Generate button
    generateBtn.addEventListener('click', async () => {
        const text = textarea.value.trim();
        
        if (text.length < 10) {
            showToast('Please enter at least 10 characters', 'error');
            return;
        }

        const maxLength = parseInt(document.getElementById('max-length').value);
        const numBeams = parseInt(document.getElementById('num-beams').value);

        await generatePLS(text, maxLength, numBeams);
    });

    // Copy button
    copyBtn.addEventListener('click', () => {
        const plsText = document.getElementById('generated-pls-text').textContent;
        copyToClipboard(plsText);
        showToast('Copied to clipboard!', 'success');
    });
}

async function generatePLS(text, maxLength = 256, numBeams = 4) {
    showLoading(true);
    hideResult('generate-result');

    try {
        const response = await fetch(API_ENDPOINTS.generate, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                technical_text: text,
                max_length: maxLength,
                num_beams: numBeams
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }

        const result = await response.json();
        displayGenerationResult(result);
        showToast('PLS generated successfully!', 'success');

    } catch (error) {
        console.error('Generation error:', error);
        showToast(error.message || 'Failed to generate PLS', 'error');
    } finally {
        showLoading(false);
    }
}

function displayGenerationResult(result) {
    const resultContainer = document.getElementById('generate-result');
    const plsText = document.getElementById('generated-pls-text');
    const generationTime = document.getElementById('generation-time');
    const numChunks = document.getElementById('num-chunks');
    const inputTokens = document.getElementById('input-tokens');
    const outputTokens = document.getElementById('output-tokens');

    // Update PLS text
    plsText.textContent = result.generated_pls || 'No output generated';

    // Update metadata
    generationTime.textContent = `${result.generation_time}s`;
    numChunks.textContent = result.num_chunks || 1;
    inputTokens.textContent = result.tokens_input || '-';
    outputTokens.textContent = result.tokens_output || '-';

    // Show result
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}

function hideResult(resultId) {
    const result = document.getElementById(resultId);
    if (result) {
        result.classList.add('hidden');
    }
}

function clearResults() {
    hideResult('classify-result');
    hideResult('generate-result');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    toast.innerHTML = `
        <span>${message}</span>
    `;

    container.appendChild(toast);

    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => {
            container.removeChild(toast);
        }, 300);
    }, 3000);
}

function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
    } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }
}

async function checkAPIHealth() {
    try {
        const response = await fetch(API_ENDPOINTS.health);
        if (response.ok) {
            const health = await response.json();
            if (!health.model_loaded) {
                showToast('Warning: Model not loaded. Some features may not work.', 'error');
            }
        }
    } catch (error) {
        console.warn('Health check failed:', error);
        showToast('Warning: Could not connect to API. Please check server status.', 'error');
    }
}

