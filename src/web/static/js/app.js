/**
 * Wildlife Monitor - Web UI Utilities
 * Vanilla JavaScript utilities for the wildlife monitoring dashboard
 */

(function() {
    'use strict';

    // ============================================
    // Date Formatting
    // ============================================

    /**
     * Format an ISO date string to a human-readable format
     * @param {string} isoString - ISO 8601 date string
     * @param {object} options - Intl.DateTimeFormat options
     * @returns {string} Formatted date string
     */
    function formatDate(isoString, options = {}) {
        if (!isoString) return 'N/A';

        try {
            const date = new Date(isoString);
            if (isNaN(date.getTime())) return 'Invalid date';

            const defaultOptions = {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            };

            return date.toLocaleDateString(undefined, { ...defaultOptions, ...options });
        } catch (e) {
            console.error('Date formatting error:', e);
            return 'Invalid date';
        }
    }

    /**
     * Format a date as relative time (e.g., "2 hours ago")
     * @param {string} isoString - ISO 8601 date string
     * @returns {string} Relative time string
     */
    function formatRelativeTime(isoString) {
        if (!isoString) return 'N/A';

        try {
            const date = new Date(isoString);
            const now = new Date();
            const diffMs = now - date;
            const diffSecs = Math.floor(diffMs / 1000);
            const diffMins = Math.floor(diffSecs / 60);
            const diffHours = Math.floor(diffMins / 60);
            const diffDays = Math.floor(diffHours / 24);

            if (diffSecs < 60) return 'Just now';
            if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
            if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
            if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;

            return formatDate(isoString, { hour: undefined, minute: undefined });
        } catch (e) {
            return 'Unknown';
        }
    }

    // ============================================
    // Confidence Formatting
    // ============================================

    /**
     * Format a confidence value as a percentage
     * @param {number} value - Confidence value (0-1)
     * @returns {string} Formatted percentage string
     */
    function formatConfidence(value) {
        if (value === null || value === undefined) return 'N/A';
        const percentage = Math.round(value * 100);
        return `${percentage}%`;
    }

    /**
     * Get CSS class for confidence level
     * @param {number} value - Confidence value (0-1)
     * @returns {string} CSS class name
     */
    function getConfidenceClass(value) {
        if (value >= 0.8) return 'high';
        if (value >= 0.5) return 'medium';
        return 'low';
    }

    // ============================================
    // Badge Classes
    // ============================================

    /**
     * Get Bootstrap badge class for trigger type
     * @param {string} type - Trigger type (motion, scheduled, manual)
     * @returns {string} Bootstrap badge classes
     */
    function getTriggerBadgeClass(type) {
        const classes = {
            'motion': 'badge trigger-motion',
            'scheduled': 'badge trigger-scheduled',
            'manual': 'badge trigger-manual'
        };
        return classes[type?.toLowerCase()] || 'badge bg-secondary';
    }

    /**
     * Get status badge class
     * @param {string} status - Status string
     * @returns {string} Status badge classes
     */
    function getStatusBadgeClass(status) {
        const classes = {
            'active': 'status-badge status-active',
            'inactive': 'status-badge status-inactive',
            'pending': 'status-badge status-pending'
        };
        return classes[status?.toLowerCase()] || 'status-badge';
    }

    // ============================================
    // Animal Icons
    // ============================================

    /**
     * Get emoji icon for animal type
     * @param {string} animalClass - Animal classification
     * @returns {string} Emoji representing the animal
     */
    function getAnimalIcon(animalClass) {
        const icons = {
            // Mammals
            'deer': '🦌',
            'fox': '🦊',
            'rabbit': '🐰',
            'squirrel': '🐿️',
            'bear': '🐻',
            'wolf': '🐺',
            'coyote': '🐕',
            'raccoon': '🦝',
            'skunk': '🦨',
            'badger': '🦡',
            'moose': '🫎',
            'elk': '🦌',
            'boar': '🐗',
            'cat': '🐱',
            'dog': '🐕',

            // Birds
            'bird': '🐦',
            'owl': '🦉',
            'hawk': '🦅',
            'eagle': '🦅',
            'turkey': '🦃',
            'duck': '🦆',
            'goose': '🪿',
            'crow': '🐦‍⬛',

            // Reptiles
            'snake': '🐍',
            'turtle': '🐢',
            'lizard': '🦎',

            // Other
            'unknown': '❓',
            'person': '🚶',
            'vehicle': '🚗'
        };

        return icons[animalClass?.toLowerCase()] || '🐾';
    }

    /**
     * Get animal badge class
     * @param {string} animalClass - Animal classification
     * @returns {string} CSS class for animal badge
     */
    function getAnimalBadgeClass(animalClass) {
        const normalized = animalClass?.toLowerCase() || 'unknown';
        return `animal-badge animal-${normalized}`;
    }

    // ============================================
    // Loading State Management
    // ============================================

    /**
     * Show loading state on an element
     * @param {HTMLElement} element - Target element
     * @param {string} text - Optional loading text
     */
    function showLoading(element, text = 'Loading...') {
        if (!element) return;

        // Store original content
        element.dataset.originalContent = element.innerHTML;
        element.classList.add('is-loading');

        // For containers, show spinner
        if (element.classList.contains('loading-target')) {
            element.innerHTML = `
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">${text}</div>
                </div>
            `;
        }
    }

    /**
     * Hide loading state on an element
     * @param {HTMLElement} element - Target element
     */
    function hideLoading(element) {
        if (!element) return;

        element.classList.remove('is-loading');

        // Restore original content if it was replaced
        if (element.dataset.originalContent && element.classList.contains('loading-target')) {
            element.innerHTML = element.dataset.originalContent;
            delete element.dataset.originalContent;
        }
    }

    // ============================================
    // API Utilities
    // ============================================

    /**
     * Wrapper around fetch with error handling
     * @param {string} endpoint - API endpoint (relative to /api/)
     * @param {object} options - Fetch options
     * @returns {Promise<object>} Parsed JSON response
     */
    async function fetchAPI(endpoint, options = {}) {
        const url = endpoint.startsWith('/') ? endpoint : `/api/${endpoint}`;

        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
            }

            // Handle empty responses
            const text = await response.text();
            return text ? JSON.parse(text) : null;
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }

    // ============================================
    // Auto-Refresh Functionality
    // ============================================

    /**
     * Auto-refresh manager for dashboard components
     */
    const AutoRefresh = {
        intervals: {},

        /**
         * Start auto-refresh for a component
         * @param {string} id - Unique identifier for this refresh
         * @param {Function} callback - Function to call on each refresh
         * @param {number} intervalMs - Refresh interval in milliseconds
         */
        start(id, callback, intervalMs = 30000) {
            this.stop(id); // Clear existing interval if any

            // Execute immediately
            callback();

            // Set up interval
            this.intervals[id] = setInterval(callback, intervalMs);
            console.log(`Auto-refresh started: ${id} (every ${intervalMs / 1000}s)`);
        },

        /**
         * Stop auto-refresh for a component
         * @param {string} id - Identifier of refresh to stop
         */
        stop(id) {
            if (this.intervals[id]) {
                clearInterval(this.intervals[id]);
                delete this.intervals[id];
                console.log(`Auto-refresh stopped: ${id}`);
            }
        },

        /**
         * Stop all auto-refresh intervals
         */
        stopAll() {
            Object.keys(this.intervals).forEach(id => this.stop(id));
        },

        /**
         * Check if auto-refresh is active for an id
         * @param {string} id - Identifier to check
         * @returns {boolean}
         */
        isActive(id) {
            return !!this.intervals[id];
        }
    };

    // Stop all refreshes when page is hidden (battery saving)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            console.log('Page hidden, pausing auto-refresh');
            // Store active intervals
            AutoRefresh._pausedIntervals = { ...AutoRefresh.intervals };
            AutoRefresh.stopAll();
        }
    });

    // ============================================
    // Toast Notifications
    // ============================================

    /**
     * Show a toast notification
     * @param {string} message - Message to display
     * @param {string} type - Type: 'success', 'error', 'warning', 'info'
     */
    function showToast(message, type = 'info') {
        // Check if Bootstrap Toast is available
        if (typeof bootstrap === 'undefined' || !bootstrap.Toast) {
            console.log(`[${type.toUpperCase()}] ${message}`);
            return;
        }

        // Create toast container if it doesn't exist
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            container.style.zIndex = '1100';
            document.body.appendChild(container);
        }

        // Create toast element
        const toastId = `toast-${Date.now()}`;
        const bgClass = {
            'success': 'bg-success',
            'error': 'bg-danger',
            'warning': 'bg-warning',
            'info': 'bg-info'
        }[type] || 'bg-secondary';

        const toastHTML = `
            <div id="${toastId}" class="toast align-items-center text-white ${bgClass}" role="alert">
                <div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        container.insertAdjacentHTML('beforeend', toastHTML);

        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
        toast.show();

        // Remove from DOM after hidden
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }

    // ============================================
    // Expose Public API
    // ============================================

    window.WildlifeMonitor = {
        // Date utilities
        formatDate,
        formatRelativeTime,

        // Formatting
        formatConfidence,
        getConfidenceClass,

        // Badge/styling helpers
        getTriggerBadgeClass,
        getStatusBadgeClass,
        getAnimalIcon,
        getAnimalBadgeClass,

        // Loading states
        showLoading,
        hideLoading,

        // API
        fetchAPI,

        // Auto-refresh
        AutoRefresh,

        // Notifications
        showToast
    };

})();
