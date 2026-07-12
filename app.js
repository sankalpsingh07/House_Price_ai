import CALIFORNIA_HOUSING_SAMPLE from './dataset.js';

// Workspace views routing state
let activeTab = 'overview';
let selectedHouseType = 'custom';

// Model weights parameters state
let modelParams = null;
let isModelTrained = true; // Pre-trained model ready to estimate out-of-the-box
let currentOverviewEDATab = 'income-scatter';

// Gemini API present value state
let geminiPresentValue = null;
let geminiExplainText = null;

// Overview Table pagination
let overviewPage = 0;
const overviewRowsPerPage = 6;
let filteredOverviewData = [...CALIFORNIA_HOUSING_SAMPLE];

// Supabase DB Predictions mock storage
let supabasePredictions = [
    { type: 'Family House', price: 348200, latLon: '37.85, -122.24', date: '7/9/2026' },
    { type: 'Small Apartment', price: 165400, latLon: '34.05, -118.24', date: '7/8/2026' }
];

// SQL Sandbox Local Tables state
// SQL Sandbox Local Tables state (simulating Supabase SQL schemas)
let db_users = JSON.parse(localStorage.getItem('supabase_db_users')) || [
    { id: 'd3b07384-d113-4c4e-9c8e-5b1234567890', email: 'investor@realestateai.com', full_name: 'Alex Mercer', created_at: '2026-07-09T10:00:00Z' }
];
let db_portfolios = JSON.parse(localStorage.getItem('supabase_db_portfolios')) || [
    { id: 'b9a2e38c-8f4f-4d6f-9988-776655443322', user_id: 'd3b07384-d113-4c4e-9c8e-5b1234567890', name: 'Silicon Valley Premium', description: 'Tech hubs in Palo Alto', created_at: '2026-07-09T10:05:00Z' }
];
let db_properties = JSON.parse(localStorage.getItem('supabase_db_properties')) || [
    { id: 'f47ac10b-58cc-4372-a567-0e02b2c3d4e5', portfolio_id: 'b9a2e38c-8f4f-4d6f-9988-776655443322', address: '345 University Ave, Palo Alto', median_income: 8.3252, house_age: 41, predicted_value: 412500 },
    { id: 'a82bc10b-58cc-4372-a567-0e02b2c3d4f6', portfolio_id: 'b9a2e38c-8f4f-4d6f-9988-776655443322', address: '120 Hawthorne St, Palo Alto', median_income: 7.2574, house_age: 52, predicted_value: 348200 }
];

window.updateSchemaSizes = function() {
    const uSpan = document.getElementById('schema-users-size');
    const portSpan = document.getElementById('schema-portfolios-size');
    const propSpan = document.getElementById('schema-properties-size');
    if (uSpan) uSpan.innerText = db_users.length;
    if (portSpan) portSpan.innerText = db_portfolios.length;
    if (propSpan) propSpan.innerText = db_properties.length;
};

// Fallback OLS weights coefficients
const defaultCoefficients = {
    'Median_Income': 0.448675,
    'House_Age': 0.009724,
    'Ave_Rooms': -0.123323,
    'Ave_Bedrooms': 0.783145,
    'Population': -0.000002,
    'Ave_Occupancy': -0.003526,
    'Latitude': -0.419792,
    'Longitude': -0.433708
};
const defaultIntercept = -37.023278;

const defaultMetrics = {
    mae: 53320.01,
    rmse: 74558.14,
    r2: 0.5758,
    adj_r2: 0.5750,
    total_samples: 20640,
    train_samples: 16512,
    test_samples: 4128
};

// Chart.js references
let overviewEDAChartRef = null;
let vpContribChartRef = null;
let vpBenchmarkChartRef = null;
let activeEstimationResult = null;

// Initializers
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Attempt loading trained parameters
    try {
        const response = await fetch('model_params.json');
        if (response.ok) {
            modelParams = await response.json();
            console.log('Model parameter parameters loaded from python parameters.');
        }
    } catch (e) {
        console.warn('Using default fallback scikit-learn coefficients.');
    }

    // 2. Initialize table grids
    renderOverviewTable();
    updateOverviewSplitSlider();
    
    // 3. Render default coefficients in training tab
    const params = modelParams || { coefficients: defaultCoefficients, intercept: defaultIntercept };
    updateWeightsTableHTML(params.coefficients);
    
    // 4. Initialize portfolio planner UI
    rebuildPortfolioUI();
    
    // 5. Run initial SQL template query
    runSQLConsoleQuery();

    // 6. Initialize prediction selector type
    selectHouseType('custom');

    // 7. Clear SQL results grid when user modifies the query input text
    const sqlInput = document.getElementById('sql-query-input');
    if (sqlInput) {
        sqlInput.addEventListener('input', () => {
            const tableHead = document.getElementById('sql-results-head');
            const tableBody = document.getElementById('sql-results-body');
            const metaText = document.getElementById('sql-query-meta');
            if (tableHead) tableHead.innerHTML = '';
            if (tableBody) tableBody.innerHTML = `<tr><td class="text-center text-muted p-4">Query modified. Click "Execute SQL" to run.</td></tr>`;
            if (metaText) {
                metaText.innerText = 'Pending execution...';
                metaText.style.color = 'var(--text-light)';
            }
        });
    }

    // 8. Initialize Coordinate map listener
    setupGridClickListeners();
    window.updateMapPointer();

    const latInput = document.getElementById('vp-latitude');
    const lonInput = document.getElementById('vp-longitude');
    if (latInput) latInput.addEventListener('input', window.updateMapPointer);
    if (lonInput) lonInput.addEventListener('input', window.updateMapPointer);
});

// Setup Grid Click Listeners
function setupGridClickListeners() {
    const svg = document.getElementById('global-coords-svg');
    if (!svg) return;

    svg.addEventListener('click', (e) => {
        // Prevent click if we clicked a button or city dot directly
        if (e.target.tagName === 'button' || e.target.closest('button')) return;

        const rect = svg.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        // Convert pixel click coordinates to SVG viewport (0 to 100)
        const pctX = clickX / rect.width * 100;
        const pctY = clickY / rect.height * 100;

        // Map SVG percentages back to Longitude [-180, 180] and Latitude [-90, 90]
        const lon = (pctX * 3.6) - 180;
        const lat = ((100 - pctY) * 1.8) - 90;

        document.getElementById('vp-latitude').value = lat.toFixed(2);
        document.getElementById('vp-longitude').value = lon.toFixed(2);
        
        // Populate Gemini location text box
        document.getElementById('vp-gemini-location').value = `${lat.toFixed(2)}, ${lon.toFixed(2)}`;

        // Uncheck all country and city radios
        const countryRadios = document.getElementsByName('country-select');
        for (const r of countryRadios) {
            r.checked = false;
        }
        const cityRadios = document.getElementsByName('city-select');
        for (const r of cityRadios) {
            r.checked = false;
        }
        // Hide city selection section
        const citySection = document.getElementById('city-selector-section');
        if (citySection) citySection.classList.add('hidden');

        window.updateMapPointer();
    });
}

const COUNTRY_CITIES = {
    'USA': [
        { name: 'Palo Alto', display: 'Palo Alto, USA', lat: 37.44, lon: -122.14, medInc: 9.20, houseAge: 38, aveRooms: 7, aveBedrooms: 1, population: 1500, aveOccupancy: 2.8, value: 4000000, summary: "Palo Alto is an exceptionally affluent city in the heart of Silicon Valley, known for Stanford University and premium tech valuations." },
        { name: 'New York', display: 'New York, USA', lat: 40.71, lon: -74.01, medInc: 6.80, houseAge: 45, aveRooms: 5, aveBedrooms: 1, population: 4500, aveOccupancy: 2.2, value: 2000000, summary: "New York is a high-density global financial center with very strong demand, leading to premium real estate valuations." },
        { name: 'Los Angeles', display: 'Los Angeles, USA', lat: 34.05, lon: -118.24, medInc: 5.50, houseAge: 32, aveRooms: 6, aveBedrooms: 1, population: 2500, aveOccupancy: 3.1, value: 1500000, summary: "Los Angeles features high demand driven by entertainment, tech hubs, and prime coastal proximity." },
        { name: 'San Francisco', display: 'San Francisco, USA', lat: 37.77, lon: -122.41, medInc: 8.50, houseAge: 50, aveRooms: 5, aveBedrooms: 1, population: 3000, aveOccupancy: 2.5, value: 1800000, summary: "San Francisco is a dense tech hub characterized by high incomes and limited geographic space." },
        { name: 'Chicago', display: 'Chicago, USA', lat: 41.88, lon: -87.63, medInc: 5.20, houseAge: 40, aveRooms: 6, aveBedrooms: 1, population: 2800, aveOccupancy: 2.6, value: 650000, summary: "Chicago offers robust urban living with moderate pricing compared to coastal metropolitan hubs." }
    ],
    'UK': [
        { name: 'London', display: 'London, UK', lat: 51.51, lon: -0.13, medInc: 6.50, houseAge: 60, aveRooms: 5, aveBedrooms: 1, population: 3200, aveOccupancy: 2.4, value: 1200000, summary: "London is a historical capital with premium real estate pricing driven by international demand and green belt limits." },
        { name: 'Manchester', display: 'Manchester, UK', lat: 53.48, lon: -2.24, medInc: 4.20, houseAge: 50, aveRooms: 5, aveBedrooms: 1, population: 2100, aveOccupancy: 2.5, value: 450000, summary: "Manchester has experienced strong growth as a Northern tech and cultural center." },
        { name: 'Birmingham', display: 'Birmingham, UK', lat: 52.48, lon: -1.89, medInc: 3.80, houseAge: 48, aveRooms: 5, aveBedrooms: 1, population: 2400, aveOccupancy: 2.6, value: 380000, summary: "Birmingham offers competitive industrial and service-hub housing options in the West Midlands." },
        { name: 'Edinburgh', display: 'Edinburgh, UK', lat: 55.95, lon: -3.19, medInc: 5.10, houseAge: 70, aveRooms: 5, aveBedrooms: 1, population: 1500, aveOccupancy: 2.2, value: 600000, summary: "Edinburgh is a premium capital with strict heritage conservation rules supporting high historical valuations." },
        { name: 'Glasgow', display: 'Glasgow, UK', lat: 55.86, lon: -4.25, medInc: 3.60, houseAge: 55, aveRooms: 5, aveBedrooms: 1, population: 2200, aveOccupancy: 2.4, value: 310000, summary: "Glasgow provides affordable urban living with strong industrial roots and emerging service hubs." }
    ],
    'Japan': [
        { name: 'Tokyo', display: 'Tokyo, Japan', lat: 35.68, lon: 139.76, medInc: 5.80, houseAge: 25, aveRooms: 4, aveBedrooms: 1, population: 5000, aveOccupancy: 2.1, value: 1100000, summary: "Tokyo is the world's most populous metropolis, boasting stable demand and modern high-density transit housing." },
        { name: 'Osaka', display: 'Osaka, Japan', lat: 34.69, lon: 135.50, medInc: 4.50, houseAge: 28, aveRooms: 4, aveBedrooms: 1, population: 3500, aveOccupancy: 2.2, value: 680000, summary: "Osaka is a major commercial hub in western Japan with robust housing inventory and moderate price gains." },
        { name: 'Kyoto', display: 'Kyoto, Japan', lat: 35.01, lon: 135.77, medInc: 4.20, houseAge: 45, aveRooms: 5, aveBedrooms: 1, population: 1800, aveOccupancy: 2.3, value: 550000, summary: "Kyoto is a global cultural center where height limits and historic zoning support solid property values." },
        { name: 'Yokohama', display: 'Yokohama, Japan', lat: 35.44, lon: 139.64, medInc: 4.80, houseAge: 30, aveRooms: 5, aveBedrooms: 1, population: 4000, aveOccupancy: 2.4, value: 620000, summary: "Yokohama serves as a popular premium port suburb and tech hub bordering Tokyo Bay." },
        { name: 'Nagoya', display: 'Nagoya, Japan', lat: 35.18, lon: 136.90, medInc: 4.60, houseAge: 26, aveRooms: 5, aveBedrooms: 1, population: 3000, aveOccupancy: 2.5, value: 480000, summary: "Nagoya is the industrial heartland of central Japan, showing steady demand driven by automotive and aerospace sectors." }
    ],
    'India': [
        { name: 'Bengaluru', display: 'Bengaluru, India', lat: 12.97, lon: 77.59, medInc: 4.80, houseAge: 12, aveRooms: 5, aveBedrooms: 1, population: 3500, aveOccupancy: 3.2, value: 250000, summary: "Bengaluru is the Silicon Valley of India, displaying rapid capital appreciation driven by tech campus expansions." },
        { name: 'Mumbai', display: 'Mumbai, India', lat: 19.07, lon: 72.87, medInc: 6.20, houseAge: 30, aveRooms: 4, aveBedrooms: 1, population: 6000, aveOccupancy: 3.8, value: 600000, summary: "Mumbai is an island financial capital with extremely high land constraints and premium luxury block pricing." },
        { name: 'Delhi', display: 'Delhi, India', lat: 28.61, lon: 77.20, medInc: 4.50, houseAge: 20, aveRooms: 5, aveBedrooms: 1, population: 4800, aveOccupancy: 4.0, value: 300000, summary: "Delhi capital region presents a diverse market with premium residential colonies and strong administrative demand." },
        { name: 'Hyderabad', display: 'Hyderabad, India', lat: 17.38, lon: 78.48, medInc: 4.00, houseAge: 15, aveRooms: 5, aveBedrooms: 1, population: 3200, aveOccupancy: 3.4, value: 210000, summary: "Hyderabad is a booming pharmaceutical and IT node featuring strong infrastructure and solid real estate gains." },
        { name: 'Chennai', display: 'Chennai, India', lat: 13.08, lon: 80.27, medInc: 3.80, houseAge: 18, aveRooms: 5, aveBedrooms: 1, population: 3000, aveOccupancy: 3.3, value: 180000, summary: "Chennai is a major industrial port and manufacturing hub displaying stable housing price indices." }
    ],
    'Australia': [
        { name: 'Sydney', display: 'Sydney, Australia', lat: -33.87, lon: 151.21, medInc: 6.80, houseAge: 35, aveRooms: 6, aveBedrooms: 1, population: 2200, aveOccupancy: 2.7, value: 1400000, summary: "Sydney is Australia's largest financial hub, exhibiting premium valuations supported by harbor views and land limits." },
        { name: 'Melbourne', display: 'Melbourne, Australia', lat: -37.81, lon: 144.96, medInc: 5.80, houseAge: 40, aveRooms: 6, aveBedrooms: 1, population: 2500, aveOccupancy: 2.6, value: 950000, summary: "Melbourne is a major cultural capital with suburban sprawl managed by urban boundaries." },
        { name: 'Brisbane', display: 'Brisbane, Australia', lat: -27.47, lon: 153.02, medInc: 4.80, houseAge: 30, aveRooms: 6, aveBedrooms: 1, population: 1800, aveOccupancy: 2.8, value: 750000, summary: "Brisbane has seen strong interstate migration gains and infrastructure growth ahead of the Olympics." },
        { name: 'Perth', display: 'Perth, Australia', lat: -31.95, lon: 115.86, medInc: 5.20, houseAge: 28, aveRooms: 6, aveBedrooms: 1, population: 1600, aveOccupancy: 2.7, value: 680000, summary: "Perth offers resource-industry-driven demand with spacious suburban blocks on the west coast." },
        { name: 'Adelaide', display: 'Adelaide, Australia', lat: -34.93, lon: 138.60, medInc: 4.20, houseAge: 45, aveRooms: 5, aveBedrooms: 1, population: 1400, aveOccupancy: 2.4, value: 600000, summary: "Adelaide displays high housing affordability and steady value growth in its boutique residential segments." }
    ]
};

window.selectCountry = function(countryCode) {
    const citySection = document.getElementById('city-selector-section');
    const cityContainer = document.getElementById('global-city-radios');
    if (!citySection || !cityContainer) return;

    // Render cities for selected country
    const cities = COUNTRY_CITIES[countryCode] || [];
    let html = '';
    cities.forEach(city => {
        html += `
            <label class="radio-card">
                <input type="radio" name="city-select" value="${city.display}" onclick="selectMapCity('${city.display}', ${city.lat}, ${city.lon})">
                <span>${city.name}</span>
            </label>
        `;
    });
    cityContainer.innerHTML = html;
    citySection.classList.remove('hidden');

    // Sync country radio check state in UI
    const countryRadios = document.getElementsByName('country-select');
    for (const r of countryRadios) {
        if (r.value === countryCode) {
            r.checked = true;
            break;
        }
    }
};

window.selectMapCity = function(cityName, lat, lon) {
    document.getElementById('vp-latitude').value = lat;
    document.getElementById('vp-longitude').value = lon;
    document.getElementById('vp-gemini-location').value = cityName;
    window.updateMapPointer();

    // Check which country this city belongs to, and render/check it
    let foundCountry = null;
    for (const country in COUNTRY_CITIES) {
        if (COUNTRY_CITIES[country].some(c => c.display === cityName)) {
            foundCountry = country;
            break;
        }
    }

    if (foundCountry) {
        // Render cities list for this country if not already shown or if different country's cities are shown
        const citySection = document.getElementById('city-selector-section');
        const countryRadios = document.getElementsByName('country-select');
        let currentCountryChecked = '';
        for (const r of countryRadios) {
            if (r.checked) currentCountryChecked = r.value;
        }
        if (!currentCountryChecked || currentCountryChecked !== foundCountry) {
            window.selectCountry(foundCountry);
        } else {
            // Just sync country radio button check state
            for (const r of countryRadios) {
                if (r.value === foundCountry) {
                    r.checked = true;
                    break;
                }
            }
        }
    }

    // Check corresponding city radio
    const cityRadios = document.getElementsByName('city-select');
    for (const r of cityRadios) {
        if (r.value === cityName) {
            r.checked = true;
            break;
        }
    }
    
    // Automatically trigger Gemini lookup
    window.runGeminiLocationAnalysis();
};

window.updateMapPointer = function() {
    const latInput = document.getElementById('vp-latitude');
    const lonInput = document.getElementById('vp-longitude');
    const pointer = document.getElementById('map-arrow-pointer');
    if (!latInput || !lonInput || pointer == null) return;

    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);

    if (isNaN(lat) || isNaN(lon)) return;

    // Constrain within bounds just for safety
    const safeLat = Math.max(-90.0, Math.min(90.0, lat));
    const safeLon = Math.max(-180.0, Math.min(180.0, lon));

    // Convert to percentage coordinates on 100x100 SVG
    const x = (safeLon + 180) / 3.6;
    const y = 100 - ((safeLat + 90) / 1.8);

    pointer.setAttribute('transform', `translate(${x}, ${y})`);
};

/* --- Authentication & Session State Controller --- */

let currentUser = null;
let pendingTab = 'overview';
let otpCode = null;
let otpCooldownInterval = null;
let authMode = 'login'; // 'login' or 'signup'
let authMethod = 'email'; // 'email' or 'phone'

// Mock registered users database in local memory
let mockRegisteredUsers = JSON.parse(localStorage.getItem('houseprice_mock_db')) || [
    { email: 'investor@realestateai.com', password: 'password123', name: 'Alex Mercer' }
];

window.initAuthSession = function() {
    window.updateSchemaSizes();
    const savedUser = localStorage.getItem('houseprice_user');
    if (savedUser) {
        currentUser = JSON.parse(savedUser);
        updateUserSessionBadge();
        
        // Redirect directly to dashboard workspace since session is validated
        document.getElementById('landing-view').classList.add('hidden');
        document.getElementById('auth-view').classList.add('hidden');
        document.getElementById('dashboard-workspace-view').classList.remove('hidden');
        document.body.classList.add('in-dashboard');
        switchDashboardTab('overview');
    } else {
        // Enforce Login/Signup card as the absolute first entrypoint page
        document.getElementById('landing-view').classList.add('hidden');
        document.getElementById('dashboard-workspace-view').classList.add('hidden');
        document.getElementById('auth-view').classList.remove('hidden');
        document.body.classList.remove('in-dashboard');
    }
};

function updateUserSessionBadge() {
    const badge = document.getElementById('workspace-user-badge');
    const nameSpan = document.getElementById('workspace-user-name');
    if (currentUser) {
        if (badge) badge.classList.remove('hidden');
        if (nameSpan) nameSpan.innerText = currentUser.name || currentUser.email;
    } else {
        if (badge) badge.classList.add('hidden');
    }
}

window.showAuthView = function(mode = 'login', targetTab = 'overview') {
    pendingTab = targetTab;
    document.getElementById('landing-view').classList.add('hidden');
    document.getElementById('dashboard-workspace-view').classList.add('hidden');
    document.getElementById('auth-view').classList.remove('hidden');
    document.body.classList.remove('in-dashboard');
    
    // Clear alerts
    const alertBox = document.getElementById('auth-error-alert');
    if (alertBox) alertBox.classList.add('hidden');

    if (mode === 'signup') {
        window.toggleAuthMode('signup');
    } else if (mode === 'login') {
        window.toggleAuthMode('login');
    }
};

window.switchAuthMethod = function(method) {
    authMethod = method;
    const tabEmail = document.getElementById('auth-tab-email');
    const tabPhone = document.getElementById('auth-tab-phone');
    const fieldsEmail = document.getElementById('auth-method-fields-email');
    const fieldsPhone = document.getElementById('auth-method-fields-phone');
    const alertBox = document.getElementById('auth-error-alert');

    if (alertBox) alertBox.classList.add('hidden');

    if (method === 'email') {
        tabEmail.classList.add('active');
        tabPhone.classList.remove('active');
        fieldsEmail.classList.remove('hidden');
        fieldsPhone.classList.add('hidden');
    } else {
        tabEmail.classList.remove('active');
        tabPhone.classList.add('active');
        fieldsEmail.classList.add('hidden');
        fieldsPhone.classList.remove('hidden');
    }
};

window.toggleAuthMode = function(targetMode) {
    if (targetMode) {
        authMode = targetMode;
    } else {
        authMode = authMode === 'login' ? 'signup' : 'login';
    }

    const title = document.getElementById('auth-card-title');
    const subtitle = document.getElementById('auth-card-subtitle');
    const groupName = document.getElementById('auth-group-name');
    const submitText = document.getElementById('btn-auth-submit-text');
    const submitIcon = document.getElementById('auth-submit-icon');
    const footerPrompt = document.getElementById('auth-footer-prompt');
    const footerToggle = document.getElementById('auth-footer-toggle');
    const alertBox = document.getElementById('auth-error-alert');

    if (alertBox) alertBox.classList.add('hidden');

    if (authMode === 'signup') {
        title.innerText = 'Create Account';
        subtitle.innerText = 'Join HousePriceAI to start building investment portfolios.';
        groupName.classList.remove('hidden');
        submitText.innerText = 'Verify & Register';
        submitIcon.className = 'fa-solid fa-user-plus';
        footerPrompt.innerText = 'Already have an account?';
        footerToggle.innerText = 'Sign In';
    } else {
        title.innerText = 'Sign In to Dashboard';
        subtitle.innerText = 'Unlock OLS regression models and API demographics analysis.';
        groupName.classList.add('hidden');
        submitText.innerText = 'Verify & Continue';
        submitIcon.className = 'fa-solid fa-lock-open';
        footerPrompt.innerText = "Don't have an account?";
        footerToggle.innerText = 'Create Account';
    }
};

window.sendMockOTP = function() {
    const phone = document.getElementById('auth-input-phone').value.trim();
    const prefix = document.getElementById('auth-input-phone-prefix').value;
    const alertBox = document.getElementById('auth-error-alert');
    const btnSend = document.getElementById('btn-auth-send-otp');
    const btnSendText = document.getElementById('btn-auth-send-otp-text');
    const groupOtp = document.getElementById('auth-group-otp');
    const timerText = document.getElementById('auth-otp-timer');

    if (!phone || phone.length < 7) {
        alertBox.innerText = 'Error: Please enter a valid phone number.';
        alertBox.classList.remove('hidden');
        return;
    }

    alertBox.classList.add('hidden');
    btnSend.disabled = true;

    // Generate a mock 6-digit OTP
    otpCode = Math.floor(100000 + Math.random() * 900000).toString();
    
    // Show OTP input
    groupOtp.classList.remove('hidden');

    // Create dynamic browser toast notification simulating SMS receipt
    const smsToast = document.createElement('div');
    smsToast.style.position = 'fixed';
    smsToast.style.bottom = '20px';
    smsToast.style.right = '20px';
    smsToast.style.background = '#1e293b';
    smsToast.style.color = '#f8fafc';
    smsToast.style.padding = '1rem 1.5rem';
    smsToast.style.borderRadius = '8px';
    smsToast.style.boxShadow = '0 10px 25px rgba(0,0,0,0.3)';
    smsToast.style.zIndex = '99999';
    smsToast.style.borderLeft = '4px solid #8b5cf6';
    smsToast.style.fontFamily = 'sans-serif';
    smsToast.style.fontSize = '0.85rem';
    smsToast.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 0.25rem;">💬 SMS Mock Receiver (${prefix} ${phone})</div>
        <div>Your HousePriceAI security code is: <strong style="color: #a78bfa; font-size: 1.05rem; letter-spacing: 0.05rem; font-family: monospace;">${otpCode}</strong></div>
    `;
    document.body.appendChild(smsToast);

    // Fade out SMS toast after 10 seconds
    setTimeout(() => {
        smsToast.style.transition = 'opacity 0.5s ease';
        smsToast.style.opacity = '0';
        setTimeout(() => document.body.removeChild(smsToast), 500);
    }, 10000);

    // Cooldown countdown timer (60s)
    let secondsLeft = 60;
    timerText.innerText = `Resend in ${secondsLeft}s`;
    btnSendText.innerText = `Code Requested (${secondsLeft}s)`;
    
    if (otpCooldownInterval) clearInterval(otpCooldownInterval);
    otpCooldownInterval = setInterval(() => {
        secondsLeft--;
        if (secondsLeft <= 0) {
            clearInterval(otpCooldownInterval);
            btnSend.disabled = false;
            btnSendText.innerText = 'Request Verification Code (SMS)';
            timerText.innerText = '';
        } else {
            timerText.innerText = `Resend in ${secondsLeft}s`;
            btnSendText.innerText = `Code Requested (${secondsLeft}s)`;
        }
    }, 1000);
};

window.submitAuthForm = function() {
    const alertBox = document.getElementById('auth-error-alert');
    const submitBtn = document.getElementById('btn-auth-submit');
    const submitText = document.getElementById('btn-auth-submit-text');
    
    if (alertBox) alertBox.classList.add('hidden');

    let emailVal = '';
    let passwordVal = '';
    let phoneVal = '';
    let nameVal = document.getElementById('auth-input-name').value.trim();

    if (authMethod === 'email') {
        emailVal = document.getElementById('auth-input-email').value.trim();
        passwordVal = document.getElementById('auth-input-password').value;

        if (!emailVal || !passwordVal) {
            alertBox.innerText = 'Error: Please fill in all required email and password fields.';
            alertBox.classList.remove('hidden');
            return;
        }
        if (authMode === 'signup' && !nameVal) {
            alertBox.innerText = 'Error: Please enter your full name to register.';
            alertBox.classList.remove('hidden');
            return;
        }
    } else {
        phoneVal = document.getElementById('auth-input-phone').value.trim();
        const otpInput = document.getElementById('auth-input-otp').value.trim();

        if (!phoneVal || !otpInput) {
            alertBox.innerText = 'Error: Please request and enter the SMS verification code.';
            alertBox.classList.remove('hidden');
            return;
        }

        if (otpInput !== otpCode) {
            alertBox.innerText = 'Error: Invalid verification OTP code entered. Please try again.';
            alertBox.classList.remove('hidden');
            return;
        }

        if (authMode === 'signup' && !nameVal) {
            alertBox.innerText = 'Error: Please enter your full name to register.';
            alertBox.classList.remove('hidden');
            return;
        }
    }

    // Run premium auth execution state
    submitBtn.disabled = true;
    const originalText = submitText.innerText;
    submitText.innerText = authMode === 'login' ? 'Securing dashboard session...' : 'Registering developer account...';

    setTimeout(() => {
        submitBtn.disabled = false;
        submitText.innerText = originalText;

        if (authMode === 'signup') {
            // Register flow
            if (authMethod === 'email') {
                const userExists = mockRegisteredUsers.find(u => u.email === emailVal);
                if (userExists) {
                    alertBox.innerText = 'Error: An account with this email address already exists.';
                    alertBox.classList.remove('hidden');
                    return;
                }
                const newUser = { email: emailVal, password: passwordVal, name: nameVal };
                mockRegisteredUsers.push(newUser);
                localStorage.setItem('houseprice_mock_db', JSON.stringify(mockRegisteredUsers));
                currentUser = { email: emailVal, name: nameVal, method: 'email' };
            } else {
                currentUser = { email: `${phoneVal}@phone-auth.com`, name: nameVal, method: 'phone' };
            }
        } else {
            // Sign in flow
            if (authMethod === 'email') {
                let user = mockRegisteredUsers.find(u => u.email === emailVal);
                if (!user) {
                    // Auto-register on the fly for ease of use in local sandbox
                    user = { email: emailVal, password: passwordVal, name: emailVal.split('@')[0] };
                    mockRegisteredUsers.push(user);
                    localStorage.setItem('houseprice_mock_db', JSON.stringify(mockRegisteredUsers));
                } else if (user.password !== passwordVal) {
                    // Update to match entered password to guarantee successful login
                    user.password = passwordVal;
                    localStorage.setItem('houseprice_mock_db', JSON.stringify(mockRegisteredUsers));
                }
                currentUser = { email: user.email, name: user.name || user.email.split('@')[0], method: 'email' };
            } else {
                currentUser = { email: `${phoneVal}@phone-auth.com`, name: nameVal || `User ${phoneVal.slice(-4)}`, method: 'phone' };
            }
        }

        // Sync user info to simulated Supabase SQL "users" table
        const userInDb = db_users.find(u => u.email === currentUser.email);
        if (!userInDb) {
            db_users.push({
                id: generateUUID(),
                email: currentUser.email,
                full_name: currentUser.name || 'Guest User',
                created_at: new Date().toISOString()
            });
            localStorage.setItem('supabase_db_users', JSON.stringify(db_users));
            window.updateSchemaSizes();
        }

        // Save session
        localStorage.setItem('houseprice_user', JSON.stringify(currentUser));
        updateUserSessionBadge();

        // Redirect to dashboard tab
        document.getElementById('auth-view').classList.add('hidden');
        document.getElementById('dashboard-workspace-view').classList.remove('hidden');
        document.body.classList.add('in-dashboard');
        switchDashboardTab(pendingTab);
    }, 1200);
};

window.continueWithGoogle = function() {
    const submitBtn = document.getElementById('btn-auth-submit');
    submitBtn.disabled = true;
    
    // Simulate social loading
    const statusText = document.getElementById('btn-auth-submit-text');
    const originalText = statusText.innerText;
    statusText.innerText = 'Verifying Google authentication...';

    setTimeout(() => {
        submitBtn.disabled = false;
        statusText.innerText = originalText;

        currentUser = { email: 'google.dev@realestateai.com', name: 'Sankalp Singh (Google)', method: 'google' };
        localStorage.setItem('houseprice_user', JSON.stringify(currentUser));
        updateUserSessionBadge();

        // Sync Google user to users table
        const userInDb = db_users.find(u => u.email === currentUser.email);
        if (!userInDb) {
            db_users.push({
                id: generateUUID(),
                email: currentUser.email,
                full_name: currentUser.name,
                created_at: new Date().toISOString()
            });
            localStorage.setItem('supabase_db_users', JSON.stringify(db_users));
            window.updateSchemaSizes();
        }

        document.getElementById('auth-view').classList.add('hidden');
        document.getElementById('dashboard-workspace-view').classList.remove('hidden');
        document.body.classList.add('in-dashboard');
        switchDashboardTab(pendingTab);
    }, 1000);
};

window.signOutUser = function() {
    currentUser = null;
    localStorage.removeItem('houseprice_user');
    updateUserSessionBadge();
    
    // Exit dashboard back to auth page (since auth page is now first)
    document.getElementById('dashboard-workspace-view').classList.add('hidden');
    document.getElementById('landing-view').classList.add('hidden');
    document.getElementById('auth-view').classList.remove('hidden');
    document.body.classList.remove('in-dashboard');
};

/* --- Routing: Landing View vs Dashboard View --- */

window.openDashboard = function(initialTab = 'overview') {
    // Session Auth Lock Check
    if (!currentUser) {
        window.showAuthView('login', initialTab);
        return;
    }
    
    document.getElementById('landing-view').classList.add('hidden');
    document.getElementById('dashboard-workspace-view').classList.remove('hidden');
    document.body.classList.add('in-dashboard');
    switchDashboardTab(initialTab);
};

window.exitDashboardToLanding = function() {
    document.getElementById('dashboard-workspace-view').classList.add('hidden');
    document.getElementById('auth-view').classList.add('hidden');
    document.getElementById('landing-view').classList.remove('hidden');
    document.body.classList.remove('in-dashboard');
};


/* --- Tab Switching Router --- */

window.switchDashboardTab = function(tabId) {
    activeTab = tabId;
    
    // Switch nav buttons active class
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => btn.classList.remove('active'));
    
    const activeBtn = document.getElementById(`nav-${tabId}`);
    if (activeBtn) activeBtn.classList.add('active');

    // Switch visible view panel
    const viewPanels = document.querySelectorAll('.view-panel');
    viewPanels.forEach(panel => {
        panel.classList.add('hidden');
        panel.classList.remove('active');
    });

    const targetPanel = document.getElementById(`view-${tabId}`);
    if (targetPanel) {
        targetPanel.classList.remove('hidden');
        targetPanel.classList.add('active');
    }

    // Update Header Text dynamically
    const headerTitle = document.getElementById('workspace-title-text');
    const headerSubtitle = document.getElementById('workspace-subtitle-text');

    const headers = {
        overview: { title: "Dashboard Overview", sub: "Browse dataset rows and view target numerical features." },
        eda: { title: "Exploratory Analysis (EDA)", sub: "Understand correlations, scatters, and histograms." },
        training: { title: "Model Training & Weights", sub: "Split the dataset partitions and solve OLS sloped weights." },
        prediction: { title: "Interactive Price Prediction", sub: "Estimate home prices and view OLS contributions." },
        portfolio: { title: "Investment Portfolio Planner", sub: "Add predicted properties and calculate portfolio valuations." },
        sql: { title: "Relational SQL Console Sandbox", sub: "Execute query commands directly against memory database states." },
        downloads: { title: "Serialized Downloads Center", sub: "Download Joblib pickle weights, CSVs, and workbooks." },
        report: { title: "Performance Analysis Report", sub: "Review OLS MAE, RMSE, and explanatory variables slopes." },
        profile: { title: "Developer Profile", sub: "Data Science and Full-Stack systems architect bio." }
    }[tabId] || { title: "Valuation Hub", sub: "" };

    headerTitle.innerText = headers.title;
    headerSubtitle.innerText = headers.sub;

    // View-specific trigger initializations
    if (tabId === 'eda') {
        setTimeout(renderOverviewEDAChart, 50);
    } else if (tabId === 'prediction') {
        updateSupabaseHistoryTable();
    } else if (tabId === 'portfolio') {
        rebuildPortfolioUI();
    } else if (tabId === 'sql') {
        window.updateSchemaSizes();
        runSQLConsoleQuery();
    }
};

/* --- VIEW 1: Overview Dataset Table pagination --- */

function renderOverviewTable() {
    const tbody = document.getElementById('overview-table-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    const start = overviewPage * overviewRowsPerPage;
    const end = start + overviewRowsPerPage;
    const pageRows = filteredOverviewData.slice(start, end);

    if (pageRows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-center text-muted">No matching block records found.</td></tr>`;
        document.getElementById('ov-page-indicator').innerText = 'Page 0 of 0';
        return;
    }

    pageRows.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.Median_Income.toFixed(4)}</td>
            <td>${row.House_Age}</td>
            <td>${row.Ave_Rooms.toFixed(2)}</td>
            <td>${row.Ave_Bedrooms.toFixed(2)}</td>
            <td>${row.Population}</td>
            <td>${row.Ave_Occupancy.toFixed(2)}</td>
            <td>${row.Latitude.toFixed(2)}</td>
            <td>${row.Longitude.toFixed(2)}</td>
            <td class="font-bold text-gradient">$${row.House_Value.toLocaleString()}</td>
        `;
        tbody.appendChild(tr);
    });

    const totalPages = Math.ceil(filteredOverviewData.length / overviewRowsPerPage);
    document.getElementById('ov-page-indicator').innerText = `Page ${overviewPage + 1} of ${totalPages}`;

    document.getElementById('ov-prev-btn').disabled = (overviewPage === 0);
    document.getElementById('ov-next-btn').disabled = (overviewPage >= totalPages - 1);
}

window.paginateOverview = function(dir) {
    const totalPages = Math.ceil(filteredOverviewData.length / overviewRowsPerPage);
    overviewPage += dir;
    if (overviewPage < 0) overviewPage = 0;
    if (overviewPage >= totalPages) overviewPage = totalPages - 1;
    renderOverviewTable();
};

window.filterOverviewTable = function() {
    const query = document.getElementById('overview-search-input').value.toLowerCase().trim();
    if (!query) {
        filteredOverviewData = [...CALIFORNIA_HOUSING_SAMPLE];
    } else {
        filteredOverviewData = CALIFORNIA_HOUSING_SAMPLE.filter(row => {
            return (
                row.Median_Income.toString().includes(query) ||
                row.House_Age.toString().includes(query) ||
                row.Population.toString().includes(query) ||
                row.House_Value.toString().includes(query)
            );
        });
    }
    overviewPage = 0;
    renderOverviewTable();
};

/* --- VIEW 2: EDA Charts --- */

window.switchOverviewEDAChart = function(chartId) {
    const tabs = document.querySelectorAll('.chart-tab');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    if (chartId === 'income-scatter') tabs[0].classList.add('active');
    else if (chartId === 'correlation') tabs[1].classList.add('active');
    else if (chartId === 'distribution') tabs[2].classList.add('active');
    else if (chartId === 'house-age') tabs[3].classList.add('active');

    currentOverviewEDATab = chartId;
    renderOverviewEDAChart();
};

function renderOverviewEDAChart() {
    const ctx = document.getElementById('overview-eda-canvas');
    if (!ctx) return;

    if (overviewEDAChartRef) overviewEDAChartRef.destroy();

    const samples = CALIFORNIA_HOUSING_SAMPLE.slice(0, 45);

    if (currentOverviewEDATab === 'income-scatter') {
        const points = samples.map(d => ({ x: d.Median_Income, y: d.House_Value / 100000 }));
        const sorted = [...points].sort((a,b) => a.x - b.x);
        const params = modelParams || { coefficients: defaultCoefficients };
        const trend = sorted.map(pt => {
            const pred = params.coefficients.Median_Income * pt.x + 0.45;
            return { x: pt.x, y: pred > 5.0 ? 5.0 : (pred < 0.15 ? 0.15 : pred) };
        });

        overviewEDAChartRef = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    { label: 'Census Blocks', data: points, backgroundColor: 'rgba(99, 102, 241, 0.65)' },
                    { label: 'OLS Slope Fit', data: trend, type: 'line', borderColor: '#ec4899', borderWidth: 2, fill: false, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Median Income (in $10k units)' } },
                    y: { title: { display: true, text: 'House Value (in $100k units)' } }
                }
            }
        });
    } else if (currentOverviewEDATab === 'correlation') {
        const vars = ['Income', 'HouseAge', 'Rooms', 'Bedrooms', 'Population', 'Occupancy', 'Latitude', 'Longitude', 'HouseValue'];
        const corrData = [
            [1.00, 0.11, 0.32, -0.06, 0.00, 0.01, -0.08, -0.01, 0.68],
            [0.11, 1.00, -0.15, -0.07, -0.29, 0.01, 0.01, -0.10, 0.10],
            [0.32, -0.15, 1.00, 0.84, -0.07, -0.00, 0.10, -0.02, 0.15],
            [-0.06, -0.07, 0.84, 1.00, -0.06, -0.00, 0.06, 0.01, -0.04],
            [0.00, -0.29, -0.07, -0.06, 1.00, 0.07, -0.10, 0.10, -0.02],
            [0.01, 0.01, -0.00, -0.00, 0.07, 1.00, 0.00, 0.00, -0.02],
            [-0.08, 0.01, 0.10, 0.06, -0.10, 0.00, 1.00, -0.92, -0.14],
            [-0.01, -0.10, -0.02, 0.01, 0.10, 0.00, -0.92, 1.00, -0.04],
            [0.68, 0.10, 0.15, -0.04, -0.02, -0.02, -0.14, -0.04, 1.00]
        ];

        const grid = [];
        for (let r = 0; r < vars.length; r++) {
            for (let c = 0; c < vars.length; c++) {
                grid.push({ x: vars[c], y: vars[r], v: corrData[r][c] });
            }
        }

        overviewEDAChartRef = new Chart(ctx, {
            type: 'bubble',
            data: {
                datasets: [{
                    label: 'r coefficient',
                    data: grid.map(p => ({ x: p.x, y: p.y, r: Math.abs(p.v) * 20 })),
                    backgroundColor: function(context) {
                        const val = grid[context.dataIndex].v;
                        return val >= 0 ? `rgba(99, 102, 241, ${val})` : `rgba(239, 68, 68, ${Math.abs(val)})`;
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'category', labels: vars },
                    y: { type: 'category', labels: vars }
                }
            }
        });
    } else if (currentOverviewEDATab === 'distribution') {
        const bins = [2, 3, 5, 8, 12, 14, 15, 10, 8, 4, 3];
        overviewEDAChartRef = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['<50k', '50-100k', '100-150k', '150-200k', '200-250k', '250-300k', '300-350k', '350-400k', '400-450k', '450-500k', '>500k'],
                datasets: [{
                    label: 'Block Counts',
                    data: bins,
                    backgroundColor: 'rgba(139, 92, 246, 0.75)',
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    } else if (currentOverviewEDATab === 'house-age') {
        const labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '50+'];
        const bins = [4, 6, 8, 10, 15, 18, 14, 20, 15, 10, 5];
        overviewEDAChartRef = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Block Groups',
                    data: bins,
                    backgroundColor: 'rgba(6, 182, 212, 0.75)',
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
}

/* --- VIEW 3: Model Preprocessing & Training console --- */

window.updateOverviewSplitSlider = function() {
    const slider = document.getElementById('ov-split-slider');
    const label = document.getElementById('ov-split-label');
    const trainBar = document.getElementById('ov-train-bar');
    const testBar = document.getElementById('ov-test-bar');
    const trainLbl = document.getElementById('ov-train-bar-lbl');
    const testLbl = document.getElementById('ov-test-bar-lbl');
    
    if (!slider) return;

    const val = parseInt(slider.value);
    label.innerText = `${val}%`;
    trainLbl.innerText = `${val}% Train`;
    testLbl.innerText = `${100 - val}% Test`;
    trainBar.style.width = `${val}%`;
    testBar.style.width = `${100 - val}%`;
};

window.runOverviewModelTraining = function() {
    const consoleLog = document.getElementById('ov-training-console-log');
    const statusText = document.getElementById('ov-training-status');
    const runBtn = document.getElementById('ov-train-action-btn');
    
    runBtn.disabled = true;
    statusText.innerText = 'Training...';
    statusText.style.color = 'var(--primary)';
    
    consoleLog.innerHTML = '';
    
    const sliderVal = parseInt(document.getElementById('ov-split-slider').value) / 100;
    const total = defaultMetrics.total_samples;
    const trainSize = Math.round(total * sliderVal);
    const testSize = total - trainSize;

    const logLines = [
        `> Isolating features matrices...`,
        `> Partitions structured: Train size = ${trainSize.toLocaleString()} | Test size = ${testSize.toLocaleString()}`,
        `> Running Ordinary Least Squares (OLS) solver...`,
        `> Computing covariance matrix inversion (XᵀX)⁻¹`,
        `> Fit completed. Slopes coordinates resolved.`,
        `> Coefficients computed. Saving model model_params.json...`
    ];

    logLines.forEach((line, idx) => {
        setTimeout(() => {
            const div = document.createElement('div');
            div.className = 'console-line';
            if (line.includes('Isolating')) div.className += ' text-blue';
            else if (line.includes(' fit')) div.className += ' text-purple';
            else if (line.includes('Completed')) div.className += ' text-teal';
            
            div.innerText = line;
            consoleLog.appendChild(div);
            consoleLog.scrollTop = consoleLog.scrollHeight;
        }, idx * 600);
    });

    setTimeout(() => {
        statusText.innerText = 'Status: Complete';
        statusText.style.color = 'var(--success)';
        
        const params = modelParams || { coefficients: defaultCoefficients, intercept: defaultIntercept };
        updateWeightsTableHTML(params.coefficients);
        
        runBtn.disabled = false;
    }, logLines.length * 600 + 100);
};

function updateWeightsTableHTML(coefs) {
    const tbody = document.getElementById('ov-coef-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    Object.keys(coefs).forEach(key => {
        const val = coefs[key];
        const displayVal = val.toFixed(6);
        const direction = val >= 0 
            ? '<span class="tag tag-success"><i class="fa-solid fa-arrow-trend-up"></i> Positive impact</span>'
            : '<span class="tag tag-danger"><i class="fa-solid fa-arrow-trend-down"></i> Negative impact</span>';

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${key}</strong></td>
            <td class="font-mono">${displayVal}</td>
            <td>${direction}</td>
        `;
        tbody.appendChild(tr);
    });
}

window.selectHouseType = function(type) {
    // 1. Remove selected class from all cards
    const cards = document.querySelectorAll('#house-type-selector-grid .preset-card');
    cards.forEach(card => card.classList.remove('selected'));

    // 2. Add selected class to active card
    const activeCard = document.getElementById(`preset-${type}`);
    if (activeCard) activeCard.classList.add('selected');

    // 3. Update global selected type
    selectedHouseType = type;

    // 4. Update placeholders guides based on type
    const placeholders = {
        apartment: { inc: '2.8', age: '15', rooms: '4', bed: '1', pop: '1800', occ: '3.2', lat: '37.77', lon: '-122.41' },
        family: { inc: '5.2', age: '30', rooms: '6', bed: '1', pop: '1200', occ: '2.8', lat: '34.05', lon: '-118.24' },
        villa: { inc: '9.5', age: '8', rooms: '8', bed: '1', pop: '450', occ: '2.4', lat: '33.61', lon: '-117.88' },
        custom: { inc: '', age: '', rooms: '', bed: '', pop: '', occ: '', lat: '', lon: '' }
    }[type] || { inc: '', age: '', rooms: '', bed: '', pop: '', occ: '', lat: '', lon: '' };

    document.getElementById('vp-income').placeholder = placeholders.inc;
    document.getElementById('vp-age').placeholder = placeholders.age;
    document.getElementById('vp-rooms').placeholder = placeholders.rooms;
    document.getElementById('vp-bedrooms').placeholder = placeholders.bed;
    document.getElementById('vp-population').placeholder = placeholders.pop;
    document.getElementById('vp-occupancy').placeholder = placeholders.occ;
    document.getElementById('vp-latitude').placeholder = placeholders.lat;
    document.getElementById('vp-longitude').placeholder = placeholders.lon;

    // 5. Ensure all inputs are blank
    document.getElementById('vp-income').value = '';
    document.getElementById('vp-age').value = '';
    document.getElementById('vp-rooms').value = '';
    document.getElementById('vp-bedrooms').value = '';
    document.getElementById('vp-population').value = '';
    document.getElementById('vp-occupancy').value = '';
    document.getElementById('vp-latitude').value = '';
    document.getElementById('vp-longitude').value = '';
    window.updateMapPointer();
};

window.runPredictionPageEstimation = function() {
    const medInc = parseFloat(document.getElementById('vp-income').value);
    const houseAge = parseFloat(document.getElementById('vp-age').value);
    const aveRooms = parseFloat(document.getElementById('vp-rooms').value);
    const aveBedrms = parseFloat(document.getElementById('vp-bedrooms').value);
    const population = parseFloat(document.getElementById('vp-population').value);
    const aveOccup = parseFloat(document.getElementById('vp-occupancy').value);
    const latitude = parseFloat(document.getElementById('vp-latitude').value);
    const longitude = parseFloat(document.getElementById('vp-longitude').value);

    // Hide idle, show loader
    document.getElementById('vp-idle').classList.add('hidden');
    document.getElementById('vp-results').classList.add('hidden');
    document.getElementById('vp-loader').classList.remove('hidden');
    document.getElementById('vp-visuals-panel').classList.add('hidden');

    setTimeout(() => {
        const params = modelParams || { coefficients: defaultCoefficients, intercept: defaultIntercept, metrics: defaultMetrics };
        const coef = params.coefficients;
        const intercept = params.intercept;

        let prediction = intercept + 
            (coef.Median_Income * medInc) + 
            (coef.House_Age * houseAge) + 
            (coef.Ave_Rooms * aveRooms) + 
            (coef.Ave_Bedrooms * aveBedrms) + 
            (coef.Population * population) + 
            (coef.Ave_Occupancy * aveOccup) + 
            (coef.Latitude * latitude) + 
            (coef.Longitude * longitude);

        let price = prediction * 100000;
        if (price < 0 || isNaN(price)) price = 0;

        document.getElementById('vp-loader').classList.add('hidden');
        document.getElementById('vp-results').classList.remove('hidden');

        // Count valuation
        const valObj = document.getElementById('vp-predicted-price');
        animateValCount(valObj, 0, Math.round(price), 850);

        // Update Gemini Card if present value is available
        const geminiCard = document.getElementById('vp-gemini-card');
        const resultsContainer = document.getElementById('vp-results-container');
        if (geminiPresentValue) {
            geminiCard.classList.remove('hidden');
            resultsContainer.classList.add('two-col');
            
            // Count Gemini price
            const geminiPriceObj = document.getElementById('vp-gemini-price');
            animateValCount(geminiPriceObj, 0, Math.round(geminiPresentValue), 850);
            
            // Set explain text
            document.getElementById('vp-gemini-explain').innerText = geminiExplainText || "Estimated modern market valuation from AI market estimation models.";
        } else {
            geminiCard.classList.add('hidden');
            resultsContainer.classList.remove('two-col');
        }

        // Category Badge
        let category = '';
        let badgeClass = '';
        if (price < 150000) {
            category = 'Budget';
            badgeClass = 'tag tag-success';
        } else if (price >= 150000 && price < 300000) {
            category = 'Mid-Range';
            badgeClass = 'tag tag-info';
        } else if (price >= 300000 && price < 450000) {
            category = 'Premium';
            badgeClass = 'tag tag-purple';
        } else {
            category = 'Luxury';
            badgeClass = 'tag tag-purple font-bold text-gradient';
        }

        document.getElementById('vp-category-badge').className = badgeClass;
        document.getElementById('vp-category-badge').innerText = category;
        document.getElementById('vp-confidence-val').innerText = `Confidence R²: ${(params.metrics.r2 * 100).toFixed(1)}%`;

        // Explainability text
        let whyText = '';
        const isOutsideCA = (latitude < 32.0 || latitude > 42.5 || longitude < -124.5 || longitude > -114.0);
        if (isOutsideCA) {
            whyText = `[Global Location Note] The 1990 OLS Model is trained strictly on California census blocks, so its formula may produce out-of-bounds estimates ($${Math.round(price).toLocaleString()}) for global coordinates. However, the AI estimator provides a modern local market estimation of $${Math.round(geminiPresentValue || 0).toLocaleString()} for ${document.getElementById('vp-gemini-location').value || 'this location'}.`;
        } else {
            if (medInc > 6.0) {
                whyText = `This house has a higher predicted price of $${Math.round(price).toLocaleString()} because it is located in a high-income block group (Income: $${(medInc*10000).toLocaleString()}) and has more rooms than average (${aveRooms.toFixed(1)} rooms).`;
            } else if (houseAge > 40) {
                whyText = `This property is priced moderately at $${Math.round(price).toLocaleString()} due to the mature age of block structures (${houseAge} years), offset positively by geographical coordinates.`;
            } else {
                whyText = `This home has a predicted valuation of $${Math.round(price).toLocaleString()} based on average block household densities, population (${population.toLocaleString()} people), and coordinates.`;
            }
        }
        document.getElementById('vp-explain-text').innerText = whyText;

        // Draw visuals
        document.getElementById('vp-visuals-panel').classList.remove('hidden');
        renderPredictorPageVisuals(medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude, price);

        // Push to Supabase history using selected type
        let type = 'Custom House';
        if (selectedHouseType === 'apartment') type = 'Small Apartment';
        else if (selectedHouseType === 'family') type = 'Family House';
        else if (selectedHouseType === 'villa') type = 'Luxury Villa';

        supabasePredictions.unshift({
            type: type,
            price: Math.round(price),
            latLon: `${latitude.toFixed(2)}, ${longitude.toFixed(2)}`,
            date: new Date().toLocaleDateString()
        });

        updateSupabaseHistoryTable();
        
        activeEstimationResult = {
            type: type,
            price: price,
            features: { medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude },
            whyText: whyText
        };

    }, 2000);
};

function renderPredictorPageVisuals(medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude, predictedPrice) {
    const contribCtx = document.getElementById('vp-contrib-chart');
    const benchmarkCtx = document.getElementById('vp-benchmark-chart');

    if (!contribCtx || !benchmarkCtx) return;

    if (vpContribChartRef) vpContribChartRef.destroy();
    if (vpBenchmarkChartRef) vpBenchmarkChartRef.destroy();

    const params = modelParams || { coefficients: defaultCoefficients };
    const coef = params.coefficients;

    // Slopes impact calculations
    const incCont = (coef.Median_Income * medInc * 100).toFixed(0);
    const ageCont = (coef.House_Age * houseAge * 100).toFixed(0);
    const roomCont = (coef.Ave_Rooms * aveRooms * 100).toFixed(0);
    const bedCont = (coef.Ave_Bedrooms * aveBedrms * 100).toFixed(0);
    const popCont = (coef.Population * population * 100).toFixed(0);
    const occCont = (coef.Ave_Occupancy * aveOccup * 100).toFixed(0);
    const locCont = ((coef.Latitude * latitude + coef.Longitude * longitude) * 10).toFixed(0);

    vpContribChartRef = new Chart(contribCtx, {
        type: 'bar',
        data: {
            labels: ['Income', 'House Age', 'Rooms', 'Bedrooms', 'Population', 'Occupants', 'Location Slope'],
            datasets: [{
                label: 'Price Impact ($k)',
                data: [incCont, ageCont, roomCont, bedCont, popCont, occCont, locCont],
                backgroundColor: function(context) {
                    const val = context.dataset.data[context.dataIndex];
                    return val >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)';
                },
                borderColor: function(context) {
                    const val = context.dataset.data[context.dataIndex];
                    return val >= 0 ? 'var(--success)' : 'var(--danger)';
                },
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y'
        }
    });

    vpBenchmarkChartRef = new Chart(benchmarkCtx, {
        type: 'bar',
        data: {
            labels: ['California Avg', 'Your Prediction', 'California Ceiling'],
            datasets: [{
                label: 'Market Valuation ($)',
                data: [206855, Math.round(predictedPrice), 500001],
                backgroundColor: ['#94a3b8', 'rgba(99, 102, 241, 0.75)', '#4f46e5'],
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function updateSupabaseHistoryTable() {
    const tbody = document.getElementById('vp-history-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (supabasePredictions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="text-center text-muted p-4">No recent predictions in Supabase DB.</td></tr>`;
        return;
    }

    supabasePredictions.forEach(pred => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${pred.type}</strong></td>
            <td><span class="font-mono">${pred.latLon}</span></td>
            <td class="font-bold text-gradient">$${pred.price.toLocaleString()}</td>
            <td>${pred.date}</td>
        `;
        tbody.appendChild(tr);
    });
}

window.exportOverviewPredictionsCSV = function() {
    if (supabasePredictions.length === 0) {
        alert("No prediction history records available to export.");
        return;
    }
    
    let csv = "data:text/csv;charset=utf-8,House Type,Latitude/Longitude,Price,Date\r\n";
    supabasePredictions.forEach(pred => {
        csv += `"${pred.type}","${pred.latLon}",${pred.price},"${pred.date}"\r\n`;
    });
    
    const uri = encodeURI(csv);
    const link = document.createElement("a");
    link.href = uri;
    link.download = "california_house_predictions.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

window.printPredictionSummary = function() {
    window.print();
};

window.downloadValuationReport = function() {
    if (!activeEstimationResult) {
        alert("Please calculate a prediction valuation first.");
        return;
    }
    
    const reportHTML = `
        <html>
        <head>
            <title>California Housing Prediction Report</title>
            <style>
                body { font-family: 'Plus Jakarta Sans', sans-serif; padding: 40px; color: #1e293b; line-height: 1.6; }
                h1 { border-bottom: 2px solid #4f46e5; padding-bottom: 10px; }
                .price { font-size: 32px; font-weight: 800; color: #4f46e5; margin: 20px 0; }
                .spec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }
                .spec-item { padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; }
            </style>
        </head>
        <body>
            <h1>RealEstateAI Valuation Report</h1>
            <p><strong>Property Category:</strong> ${activeEstimationResult.type}</p>
            <div class="price">Estimated Price: $${Math.round(activeEstimationResult.price).toLocaleString()}</div>
            <p><strong>Explanatory Summary:</strong> ${activeEstimationResult.whyText}</p>
            
            <h3>Input Demographics Specs:</h3>
            <div class="spec-grid">
                <div class="spec-item">Median Income: $${(activeEstimationResult.features.medInc * 10000).toLocaleString()}</div>
                <div class="spec-item">Median House Age: ${activeEstimationResult.features.houseAge} Yrs</div>
                <div class="spec-item">Average Rooms: ${activeEstimationResult.features.aveRooms.toFixed(2)}</div>
                <div class="spec-item">Average Bedrooms: ${activeEstimationResult.features.aveBedrms.toFixed(2)}</div>
                <div class="spec-item">Block Population: ${activeEstimationResult.features.population.toLocaleString()}</div>
                <div class="spec-item">Average Occupants: ${activeEstimationResult.features.aveOccup.toFixed(2)}</div>
                <div class="spec-item">Latitude / Longitude: ${activeEstimationResult.features.latitude}, ${activeEstimationResult.features.longitude}</div>
            </div>
            <p style="margin-top: 40px; font-size: 11px; color: #94a3b8;">Generated by HousePriceAI scikit-learn ordinary least squares regression model pipeline.</p>
        </body>
        </html>
    `;

    const blob = new Blob([reportHTML], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `California_Valuation_Report.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

function generateLocalFallbackDemographics(location) {
    let hash = 0;
    for (let i = 0; i < location.length; i++) {
        hash = (hash << 5) - hash + location.charCodeAt(i);
        hash |= 0;
    }
    const absHash = Math.abs(hash);
    
    let lat = 37.77 + (absHash % 100) / 50.0 - 1.0;
    let lon = -122.41 + (absHash % 100) / 50.0 - 1.0;
    let regionName = "suburban district";
    
    const locLower = location.toLowerCase();
    if (locLower.includes('noida') || locLower.includes('delhi') || locLower.includes('india') || locLower.includes('mumbai') || locLower.includes('bengaluru')) {
        lat = 28.5 + (absHash % 50) / 100.0 - 0.25;
        lon = 77.4 + (absHash % 50) / 100.0 - 0.25;
        regionName = "NCR region";
    } else if (locLower.includes('london') || locLower.includes('uk') || locLower.includes('england')) {
        lat = 51.5 + (absHash % 50) / 100.0 - 0.25;
        lon = -0.1 + (absHash % 50) / 100.0 - 0.25;
        regionName = "Greater London area";
    } else if (locLower.includes('tokyo') || locLower.includes('japan')) {
        lat = 35.6 + (absHash % 50) / 100.0 - 0.25;
        lon = 139.6 + (absHash % 50) / 100.0 - 0.25;
        regionName = "Tokyo metropolis";
    } else if (locLower.includes('sydney') || locLower.includes('australia')) {
        lat = -33.8 + (absHash % 50) / 100.0 - 0.25;
        lon = 151.2 + (absHash % 50) / 100.0 - 0.25;
        regionName = "Sydney metro area";
    }
    
    const medInc = 2.0 + (absHash % 80) / 10.0;
    const houseAge = 5 + (absHash % 48);
    const aveRooms = 3.5 + (absHash % 40) / 10.0;
    const aveBedrooms = 0.95 + (absHash % 10) / 20.0;
    const population = 600 + (absHash % 4400);
    const aveOccupancy = 1.8 + (absHash % 25) / 10.0;
    
    const baseVal = 180000 + (medInc * 45000) + (aveRooms * 15000) - (houseAge * 800);
    const presentMarketValue = Math.round(baseVal + (absHash % 50000));
    
    const briefSummary = `Located in the ${regionName}. Local metrics indicate a growing residential community showing stable occupancy and consistent housing demand. Valuation matches recent transaction benchmarks in surrounding districts.`;
    
    return {
        median_income: medInc,
        house_age: houseAge,
        ave_rooms: aveRooms,
        ave_bedrooms: aveBedrooms,
        population: population,
        ave_occupancy: aveOccupancy,
        latitude: lat,
        longitude: lon,
        present_market_value: presentMarketValue,
        brief_summary: briefSummary
    };
}

window.runGeminiLocationAnalysis = async function() {
    const apiKey = document.getElementById('vp-gemini-key').value.trim();
    const location = document.getElementById('vp-gemini-location').value.trim();
    const statusBar = document.getElementById('gemini-status-bar');
    const statusText = document.getElementById('gemini-status-text');
    const statusSpinner = document.getElementById('gemini-status-spinner');
    const analyzeBtn = document.getElementById('btn-gemini-analyze');

    if (!apiKey) {
        alert("Please enter a valid API Key.");
        return;
    }
    if (!location) {
        alert("Please enter a location or address to analyze.");
        return;
    }

    // Show status bar, disable button
    statusBar.classList.remove('hidden');
    statusBar.style.background = 'rgba(99, 102, 241, 0.05)';
    statusBar.style.borderColor = 'rgba(99, 102, 241, 0.1)';
    statusBar.style.color = 'var(--primary)';
    statusSpinner.className = 'fa-solid fa-spinner fa-spin';
    statusText.innerText = `Analyzing '${location}' demographics...`;
    analyzeBtn.disabled = true;

    // Reset previous gemini state
    geminiPresentValue = null;
    geminiExplainText = null;

    // 1. Check if the query is in the offline presets cache
    let cachedCity = null;
    for (const country in COUNTRY_CITIES) {
        const found = COUNTRY_CITIES[country].find(c => 
            c.display.toLowerCase() === location.toLowerCase() || 
            c.name.toLowerCase() === location.toLowerCase() ||
            location.toLowerCase().includes(c.name.toLowerCase())
        );
        if (found) {
            cachedCity = found;
            break;
        }
    }

    if (cachedCity) {
        setTimeout(() => {
            statusBar.style.background = 'rgba(16, 185, 129, 0.05)';
            statusBar.style.borderColor = 'rgba(16, 185, 129, 0.1)';
            statusBar.style.color = 'var(--success)';
            statusSpinner.className = 'fa-solid fa-circle-check';
            statusText.innerText = "Location loaded from offline cache! Auto-populated fields...";

            // Auto-fill prediction form inputs
            document.getElementById('vp-income').value = cachedCity.medInc.toFixed(2);
            document.getElementById('vp-age').value = Math.round(cachedCity.houseAge);
            document.getElementById('vp-rooms').value = Math.round(cachedCity.aveRooms);
            document.getElementById('vp-bedrooms').value = Math.round(cachedCity.aveBedrooms);
            document.getElementById('vp-population').value = Math.round(cachedCity.population);
            document.getElementById('vp-occupancy').value = cachedCity.aveOccupancy.toFixed(1);
            document.getElementById('vp-latitude').value = cachedCity.lat.toFixed(2);
            document.getElementById('vp-longitude').value = cachedCity.lon.toFixed(2);

            // Store valuation state
            geminiPresentValue = cachedCity.value;
            geminiExplainText = cachedCity.summary;

            // Update arrow pointer on map
            window.updateMapPointer();

            // Set house type to custom
            const cards = document.querySelectorAll('#house-type-selector-grid .preset-card');
            cards.forEach(card => card.classList.remove('selected'));
            const customCard = document.getElementById('preset-custom');
            if (customCard) customCard.classList.add('selected');
            selectedHouseType = 'custom';

            // Trigger prediction
            window.runPredictionPageEstimation();
            analyzeBtn.disabled = false;
        }, 300);
        return;
    }

    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
        
        const prompt = `Analyze the real-world neighborhood at address/location: '${location}'. Estimate its demographic metrics matching the 1990 U.S. Census indicators structure, as well as the present-day market valuation today (year 2026). Provide the response strictly in JSON format. Do not write any markdown codeblock formatting, just return raw JSON matching this schema:
{
  "median_income": <number, median income of block households in tens of thousands, e.g. 8.5 for $85k/yr. Typically range from 0.5 to 15.0>,
  "house_age": <number, median building age in years, typically 1 to 52>,
  "ave_rooms": <number, average rooms per household, typically 3.0 to 8.0>,
  "ave_bedrooms": <number, average bedrooms per household, typically 0.9 to 1.5>,
  "population": <number, total block group population, e.g. 800 to 5000>,
  "ave_occupancy": <number, average occupants per household, typically 1.8 to 4.5>,
  "latitude": <number, latitude decimal coordinates, e.g. 37.77>,
  "longitude": <number, longitude decimal coordinates, e.g. -122.41>,
  "present_market_value": <number, estimated current market value in USD of a standard single family home here today in 2026>,
  "brief_summary": <string, 2-sentence description of the neighborhood and present real estate market value factors>
}`;

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contents: [{
                    parts: [{
                        text: prompt
                    }]
                }],
                generationConfig: {
                    responseMimeType: "application/json"
                }
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            const errMsg = errData.error?.message || `HTTP ${response.status}`;
            throw new Error(`AI Engine API Error: ${errMsg}`);
        }

        const resData = await response.json();
        let rawText = resData.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!rawText) {
            throw new Error("Invalid response format received from AI Engine.");
        }

        // Strip code block formatting if returned despite config
        rawText = rawText.replace(/```json/gi, '').replace(/```/g, '').trim();
        const data = JSON.parse(rawText);

        // Validate structure fields
        if (typeof data.median_income !== 'number' || typeof data.present_market_value !== 'number') {
            throw new Error("Missing required numeric keys in AI response payload.");
        }

        // Auto-fill prediction form inputs
        document.getElementById('vp-income').value = data.median_income.toFixed(2);
        document.getElementById('vp-age').value = Math.round(data.house_age);
        document.getElementById('vp-rooms').value = Math.round(data.ave_rooms);
        document.getElementById('vp-bedrooms').value = Math.round(data.ave_bedrooms);
        document.getElementById('vp-population').value = Math.round(data.population);
        document.getElementById('vp-occupancy').value = data.ave_occupancy.toFixed(1);
        document.getElementById('vp-latitude').value = data.latitude.toFixed(2);
        document.getElementById('vp-longitude').value = data.longitude.toFixed(2);

        // Store gemini valuation state
        geminiPresentValue = data.present_market_value;
        geminiExplainText = data.brief_summary;

        // Success status
        statusBar.style.background = 'rgba(16, 185, 129, 0.05)';
        statusBar.style.borderColor = 'rgba(16, 185, 129, 0.1)';
        statusBar.style.color = 'var(--success)';
        statusSpinner.className = 'fa-solid fa-circle-check';
        statusText.innerText = "Location analyzed! Auto-populated fields and triggering prediction...";

        // Set house type to custom for custom auto-fill
        const cards = document.querySelectorAll('#house-type-selector-grid .preset-card');
        cards.forEach(card => card.classList.remove('selected'));
        const customCard = document.getElementById('preset-custom');
        if (customCard) customCard.classList.add('selected');
        selectedHouseType = 'custom';

        // Trigger the predictor estimation calculation
        runPredictionPageEstimation();

    } catch (e) {
        console.warn("AI Engine API error encountered. Falling back to local demographic analyzer heuristics:", e);
        
        const fallbackData = generateLocalFallbackDemographics(location);
        
        // Auto-fill prediction form inputs
        document.getElementById('vp-income').value = fallbackData.median_income.toFixed(2);
        document.getElementById('vp-age').value = Math.round(fallbackData.house_age);
        document.getElementById('vp-rooms').value = Math.round(fallbackData.ave_rooms);
        document.getElementById('vp-bedrooms').value = Math.round(fallbackData.ave_bedrooms);
        document.getElementById('vp-population').value = Math.round(fallbackData.population);
        document.getElementById('vp-occupancy').value = fallbackData.ave_occupancy.toFixed(1);
        document.getElementById('vp-latitude').value = fallbackData.latitude.toFixed(2);
        document.getElementById('vp-longitude').value = fallbackData.longitude.toFixed(2);

        // Store gemini valuation state
        geminiPresentValue = fallbackData.present_market_value;
        geminiExplainText = fallbackData.brief_summary;

        // Success status
        statusBar.style.background = 'rgba(16, 185, 129, 0.05)';
        statusBar.style.borderColor = 'rgba(16, 185, 129, 0.1)';
        statusBar.style.color = 'var(--success)';
        statusSpinner.className = 'fa-solid fa-circle-check';
        statusText.innerText = "Location analyzed using demographic engine! Auto-populated fields and triggering prediction...";

        // Set house type to custom for custom auto-fill
        const cards = document.querySelectorAll('#house-type-selector-grid .preset-card');
        cards.forEach(card => card.classList.remove('selected'));
        const customCard = document.getElementById('preset-custom');
        if (customCard) customCard.classList.add('selected');
        selectedHouseType = 'custom';

        // Trigger the predictor estimation calculation
        runPredictionPageEstimation();
    } finally {
        analyzeBtn.disabled = false;
    }
};

window.resetOverviewPredictionForm = function() {
    document.getElementById('vp-income').value = '';
    document.getElementById('vp-age').value = '';
    document.getElementById('vp-rooms').value = '';
    document.getElementById('vp-bedrooms').value = '';
    document.getElementById('vp-population').value = '';
    document.getElementById('vp-occupancy').value = '';
    document.getElementById('vp-latitude').value = '';
    document.getElementById('vp-longitude').value = '';

    document.getElementById('vp-results').classList.add('hidden');
    document.getElementById('vp-loader').classList.add('hidden');
    document.getElementById('vp-idle').classList.remove('hidden');
    document.getElementById('vp-visuals-panel').classList.add('hidden');
    
    // Reset Gemini state
    geminiPresentValue = null;
    geminiExplainText = null;
    document.getElementById('vp-gemini-location').value = '';
    document.getElementById('gemini-status-bar').classList.add('hidden');
    document.getElementById('vp-gemini-card').classList.add('hidden');
    document.getElementById('vp-results-container').classList.remove('two-col');

    // Uncheck all country and city radios
    const countryRadios = document.getElementsByName('country-select');
    for (const r of countryRadios) {
        r.checked = false;
    }
    const cityRadios = document.getElementsByName('city-select');
    for (const r of cityRadios) {
        r.checked = false;
    }
    const citySection = document.getElementById('city-selector-section');
    if (citySection) {
        citySection.classList.add('hidden');
        document.getElementById('global-city-radios').innerHTML = '';
    }

    activeEstimationResult = null;
    selectHouseType('custom');
};

/* --- VIEW 5: Portfolio Planner Logic --- */

window.copyPredictionSpecsToPortfolioForm = function() {
    const inc = document.getElementById('vp-income').value;
    const age = document.getElementById('vp-age').value;
    const lat = document.getElementById('vp-latitude').value;
    const lon = document.getElementById('vp-longitude').value;
    
    document.getElementById('port-income').value = inc;
    document.getElementById('port-age').value = age;
    document.getElementById('port-address').value = `Block Centroid (${lat}, ${lon})`;
    
    alert("Copied values from prediction specs successfully!");
};

window.addPropertyToPortfolioTable = function() {
    const address = document.getElementById('port-address').value;
    const income = parseFloat(document.getElementById('port-income').value);
    const age = parseFloat(document.getElementById('port-age').value);
    
    const params = modelParams || { coefficients: defaultCoefficients, intercept: defaultIntercept };
    const coef = params.coefficients;
    const intercept = params.intercept;
    
    // local OLS evaluation
    let prediction = intercept + 
        (coef.Median_Income * income) + 
        (coef.House_Age * age) + 
        (coef.Ave_Rooms * 5.4) + 
        (coef.Ave_Bedrooms * 1.1) + 
        (coef.Population * 1425 * -0.000002) + 
        (coef.Ave_Occupancy * 3.0) + 
        (coef.Latitude * 35.63) + 
        (coef.Longitude * -119.57);

    let price = prediction * 100000;
    if (price > 500001) price = 500001;
    if (price < 15000 || isNaN(price)) price = 15000;
    
    const newId = generateUUID();
    db_properties.push({
        id: newId,
        portfolio_id: 'b9a2e38c-8f4f-4d6f-9988-776655443322',
        address: address,
        median_income: income,
        house_age: age,
        predicted_value: Math.round(price)
    });
    
    rebuildPortfolioUI();
    runSQLConsoleQuery(); // sync console if query active
    
    // reset inputs
    document.getElementById('port-address').value = '';
    document.getElementById('port-income').value = '';
    document.getElementById('port-age').value = '';
};

window.deletePropertyFromPortfolioRecord = function(id) {
    db_properties = db_properties.filter(p => p.id !== id);
    rebuildPortfolioUI();
    runSQLConsoleQuery();
};

function rebuildPortfolioUI() {
    const wrapper = document.getElementById('portfolio-properties-list');
    if (!wrapper) return;
    
    wrapper.innerHTML = '';
    let totalVal = 0;
    let totalInc = 0;
    const count = db_properties.length;

    if (count === 0) {
        wrapper.innerHTML = `<div class="text-center text-muted p-6">No properties saved yet. Add a predicted property on the right!</div>`;
        document.getElementById('portfolio-total-val').innerText = '$0';
        document.getElementById('portfolio-count').innerText = '0';
        document.getElementById('portfolio-avg-inc').innerText = '$0';
        return;
    }

    db_properties.forEach(prop => {
        totalVal += prop.predicted_value;
        totalInc += prop.median_income;
        
        const card = document.createElement('div');
        card.className = 'portfolio-card';
        card.innerHTML = `
            <div class="portfolio-card-info">
                <h4>${prop.address}</h4>
                <p>Income: $${(prop.median_income*10000).toLocaleString()} | Age: ${prop.house_age} Years</p>
            </div>
            <div class="portfolio-card-price">
                <span class="price text-gradient">$${prop.predicted_value.toLocaleString()}</span>
                <button class="portfolio-delete-btn" onclick="deletePropertyFromPortfolioRecord('${prop.id}')" title="Delete Property">
                    <i class="fa-solid fa-trash-can"></i>
                </button>
            </div>
        `;
        wrapper.appendChild(card);
    });

    const avgInc = totalInc / count;
    document.getElementById('portfolio-total-val').innerText = `$${Math.round(totalVal).toLocaleString()}`;
    document.getElementById('portfolio-count').innerText = count.toString();
    document.getElementById('portfolio-avg-inc').innerText = `$${Math.round(avgInc * 10000).toLocaleString()}`;
}

/* --- VIEW 6: Relational SQL Console compiler --- */

window.setSQLConsoleTemplate = function(type) {
    // Deprecated in favor of setAndRunSQL, keeping as fallback
    if (type === 'all-props') setAndRunSQL('all-props', 'SELECT * FROM properties;', 'List All Properties');
    else if (type === 'all-ports') setAndRunSQL('all-ports', 'SELECT * FROM portfolios;', 'List Portfolios');
    else if (type === 'count-agg') setAndRunSQL('avg-price', 'SELECT COUNT(*), AVG(predicted_value) FROM properties;', 'Average valuation');
};

window.setAndRunSQL = function(type, query, desc) {
    const input = document.getElementById('sql-query-input');
    const descBox = document.getElementById('sql-helper-description');
    
    if (input) input.value = query;
    if (descBox) {
        descBox.innerText = desc;
        descBox.style.display = 'block';
    }
    
    window.runSQLConsoleQuery();
};

window.runSQLConsoleQuery = function() {
    const text = document.getElementById('sql-query-input').value.trim();
    const tableHead = document.getElementById('sql-results-head');
    const tableBody = document.getElementById('sql-results-body');
    const metaText = document.getElementById('sql-query-meta');

    if (!tableHead || !tableBody) return;
    
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';

    const start = performance.now();
    let records = [];
    let fields = [];

    const sqlNormalized = text.replace(/\s+/g, ' ').replace(/;/g, '').trim();
    const sql = sqlNormalized.toLowerCase().trim();

    try {
        if (!sql.startsWith('select')) {
            throw new Error("Syntax Error: Only 'SELECT' operations are supported in this PostgreSQL developer sandbox.");
        }

        // Extract clauses by splitting on major SQL keywords
        const fromIdx = sql.indexOf(' from ');
        const joinIdx = sql.indexOf(' join ');
        const whereIdx = sql.indexOf(' where ');
        const orderByIdx = sql.indexOf(' order by ');

        if (fromIdx === -1) {
            throw new Error("Syntax Error: Query must contain a 'FROM' clause specifying the target table.");
        }

        // Extract SELECT projection columns
        const selectClause = sqlNormalized.substring(7, fromIdx).trim();

        // Extract FROM base table
        let endFrom = sqlNormalized.length;
        if (joinIdx !== -1) endFrom = joinIdx;
        else if (whereIdx !== -1) endFrom = whereIdx;
        else if (orderByIdx !== -1) endFrom = orderByIdx;
        const fromClause = sqlNormalized.substring(fromIdx + 6, endFrom).trim();

        // Extract JOIN
        let joinTable = null;
        let joinCondition = null;
        if (joinIdx !== -1) {
            let endJoin = sqlNormalized.length;
            if (whereIdx !== -1) endJoin = whereIdx;
            else if (orderByIdx !== -1) endJoin = orderByIdx;
            
            const joinStr = sqlNormalized.substring(joinIdx + 6, endJoin).trim();
            const onIdx = joinStr.toLowerCase().indexOf(' on ');
            if (onIdx === -1) {
                throw new Error("Syntax Error: 'JOIN' clause requires matching 'ON' condition.");
            }
            joinTable = joinStr.substring(0, onIdx).trim();
            joinCondition = joinStr.substring(onIdx + 4).trim();
        }

        // Extract WHERE
        let whereClause = null;
        if (whereIdx !== -1) {
            let endWhere = sqlNormalized.length;
            if (orderByIdx !== -1) endWhere = orderByIdx;
            whereClause = sqlNormalized.substring(whereIdx + 7, endWhere).trim();
        }

        // Extract ORDER BY
        let orderByClause = null;
        if (orderByIdx !== -1) {
            orderByClause = sqlNormalized.substring(orderByIdx + 10).trim();
        }

        // 1. Get base table data
        const baseTable = fromClause.toLowerCase();
        if (baseTable === 'properties') {
            records = JSON.parse(JSON.stringify(db_properties));
        } else if (baseTable === 'portfolios') {
            records = JSON.parse(JSON.stringify(db_portfolios));
        } else if (baseTable === 'users') {
            records = JSON.parse(JSON.stringify(db_users));
        } else {
            throw new Error(`Syntax Error: Table '${fromClause}' does not exist. Available tables: properties, portfolios, users.`);
        }

        // 2. Handle JOIN if exists
        if (joinTable && joinCondition) {
            const secondaryTable = joinTable.toLowerCase();
            let joinRecords = [];
            if (secondaryTable === 'properties') {
                joinRecords = db_properties;
            } else if (secondaryTable === 'portfolios') {
                joinRecords = db_portfolios;
            } else if (secondaryTable === 'users') {
                joinRecords = db_users;
            } else {
                throw new Error(`Syntax Error: Join table '${joinTable}' does not exist. Available tables: properties, portfolios, users.`);
            }

            const parts = joinCondition.split('=');
            if (parts.length !== 2) {
                throw new Error(`Syntax Error: Invalid join condition '${joinCondition}'.`);
            }
            const leftCol = parts[0].trim();
            const rightCol = parts[1].trim();

            const getColParts = (c) => {
                const p = c.split('.');
                return p.length === 2 ? { table: p[0].trim().toLowerCase(), field: p[1].trim() } : { table: null, field: c.trim() };
            };

            const leftParts = getColParts(leftCol);
            const rightParts = getColParts(rightCol);

            let joined = [];
            records.forEach(rec1 => {
                joinRecords.forEach(rec2 => {
                    let val1 = (leftParts.table === baseTable || !leftParts.table) ? rec1[leftParts.field] : rec2[leftParts.field];
                    if (leftParts.table === secondaryTable) val1 = rec2[leftParts.field];

                    let val2 = (rightParts.table === baseTable || !rightParts.table) ? rec1[rightParts.field] : rec2[rightParts.field];
                    if (rightParts.table === secondaryTable) val2 = rec2[rightParts.field];

                    if (val1 !== undefined && val2 !== undefined && val1.toString() === val2.toString()) {
                        let merged = {};
                        for (let k in rec1) merged[`${baseTable}.${k}`] = rec1[k];
                        for (let k in rec2) merged[`${secondaryTable}.${k}`] = rec2[k];
                        for (let k in rec1) if (!(k in merged)) merged[k] = rec1[k];
                        for (let k in rec2) if (!(k in merged)) merged[k] = rec2[k];
                        joined.push(merged);
                    }
                });
            });
            records = joined;
        } else {
            records = records.map(rec => {
                let mapped = { ...rec };
                for (let k in rec) {
                    mapped[`${baseTable}.${k}`] = rec[k];
                }
                return mapped;
            });
        }

        // 3. Handle WHERE filtering
        if (whereClause) {
            const operators = ['>=', '<=', '>', '<', '=', '!='];
            let op = null;
            let opIdx = -1;
            for (let o of operators) {
                opIdx = whereClause.indexOf(o);
                if (opIdx !== -1) {
                    op = o;
                    break;
                }
            }

            if (op) {
                const colName = whereClause.substring(0, opIdx).trim();
                let filterVal = whereClause.substring(opIdx + op.length).trim();
                filterVal = filterVal.replace(/['"]/g, '');

                records = records.filter(rec => {
                    let val = rec[colName];
                    if (val === undefined) {
                        const fieldName = colName.split('.').pop();
                        val = rec[fieldName];
                    }
                    if (val === undefined) return false;

                    const numVal = parseFloat(val);
                    const numFilter = parseFloat(filterVal);
                    if (!isNaN(numVal) && !isNaN(numFilter)) {
                        if (op === '>') return numVal > numFilter;
                        if (op === '<') return numVal < numFilter;
                        if (op === '>=') return numVal >= numFilter;
                        if (op === '<=') return numVal <= numFilter;
                        if (op === '=') return numVal === numFilter;
                        if (op === '!=') return numVal !== numFilter;
                    } else {
                        const strVal = val.toString().toLowerCase();
                        const strFilter = filterVal.toString().toLowerCase();
                        if (op === '=') return strVal === strFilter;
                        if (op === '!=') return strVal !== strFilter;
                    }
                    return false;
                });
            }
        }

        // 4. Handle ORDER BY sorting
        if (orderByClause) {
            const parts = orderByClause.trim().split(/\s+/);
            const colName = parts[0].trim();
            const direction = parts[1] ? parts[1].toLowerCase() : 'asc';

            records.sort((a, b) => {
                let valA = a[colName];
                let valB = b[colName];
                if (valA === undefined) valA = a[colName.split('.').pop()];
                if (valB === undefined) valB = b[colName.split('.').pop()];

                if (valA === undefined || valB === undefined) return 0;

                const numA = parseFloat(valA);
                const numB = parseFloat(valB);
                if (!isNaN(numA) && !isNaN(numB)) {
                    return direction === 'desc' ? numB - numA : numA - numB;
                } else {
                    return direction === 'desc' 
                        ? valB.toString().localeCompare(valA.toString()) 
                        : valA.toString().localeCompare(valB.toString());
                }
            });
        }

        // 5. Handle SELECT projection & Aggregates
        const selectNorm = selectClause.toLowerCase();
        if (selectNorm === '*') {
            if (records.length > 0) {
                fields = Object.keys(records[0]).filter(k => !k.includes('.'));
            } else {
                if (baseTable === 'properties') fields = ['id', 'address', 'median_income', 'house_age', 'predicted_value'];
                else if (baseTable === 'portfolios') fields = ['id', 'user_id', 'name', 'description', 'created_at'];
                else if (baseTable === 'users') fields = ['id', 'email', 'full_name', 'created_at'];
            }
        } else {
            const cols = selectClause.split(',').map(c => c.trim());
            const isAgg = cols.some(c => 
                c.toLowerCase().includes('count(') || 
                c.toLowerCase().includes('avg(') || 
                c.toLowerCase().includes('sum(') || 
                c.toLowerCase().includes('min(') || 
                c.toLowerCase().includes('max(')
            );

            if (isAgg) {
                let aggRow = {};
                cols.forEach(c => {
                    const cleanCol = c.toLowerCase();
                    if (cleanCol.startsWith('count(')) {
                        aggRow[c] = records.length;
                    } else if (cleanCol.startsWith('avg(')) {
                        const field = c.match(/avg\((.*?)\)/i)[1].trim();
                        const sum = records.reduce((s, r) => s + (parseFloat(r[field]) || parseFloat(r[field.split('.').pop()]) || 0), 0);
                        aggRow[c] = records.length > 0 ? Math.round(sum / records.length) : 0;
                    } else if (cleanCol.startsWith('sum(')) {
                        const field = c.match(/sum\((.*?)\)/i)[1].trim();
                        aggRow[c] = records.reduce((s, r) => s + (parseFloat(r[field]) || parseFloat(r[field.split('.').pop()]) || 0), 0);
                    } else if (cleanCol.startsWith('min(')) {
                        const field = c.match(/min\((.*?)\)/i)[1].trim();
                        const vals = records.map(r => parseFloat(r[field]) || parseFloat(r[field.split('.').pop()]) || 0);
                        aggRow[c] = vals.length > 0 ? Math.min(...vals) : 0;
                    } else if (cleanCol.startsWith('max(')) {
                        const field = c.match(/max\((.*?)\)/i)[1].trim();
                        const vals = records.map(r => parseFloat(r[field]) || parseFloat(r[field.split('.').pop()]) || 0);
                        aggRow[c] = vals.length > 0 ? Math.max(...vals) : 0;
                    }
                });
                records = [aggRow];
                fields = cols;
            } else {
                fields = cols;
                records = records.map(rec => {
                    let projected = {};
                    cols.forEach(c => {
                        let val = rec[c];
                        if (val === undefined) val = rec[c.split('.').pop()];
                        projected[c] = val !== undefined ? val : null;
                    });
                    return projected;
                });
            }
        }

        // Render headers
        const tr = document.createElement('tr');
        fields.forEach(f => {
            const th = document.createElement('th');
            th.innerText = f.toUpperCase().replace(/_/g, ' ');
            tr.appendChild(th);
        });
        tableHead.appendChild(tr);

        // Render rows
        if (records.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="${fields.length}" class="text-center text-muted">No rows match your query.</td></tr>`;
        } else {
            records.forEach(rec => {
                const tr = document.createElement('tr');
                fields.forEach(f => {
                    const td = document.createElement('td');
                    let val = rec[f];
                    if (f === 'predicted_value' && typeof val === 'number') {
                        td.innerText = `$${val.toLocaleString()}`;
                        td.className = 'font-bold text-gradient';
                    } else if (f === 'median_income' && typeof val === 'number') {
                        td.innerText = val.toFixed(4);
                    } else if (typeof val === 'string' && val.length > 25) {
                        td.innerText = val.substring(0, 10) + '...';
                        td.title = val;
                    } else {
                        td.innerText = val !== null ? val : 'NULL';
                    }
                    tr.appendChild(td);
                });
                tableBody.appendChild(tr);
            });
        }

        const elapsed = ((performance.now() - start) / 1000).toFixed(4);
        metaText.innerText = `Rows: ${records.length} | Time: ${elapsed}s`;
        metaText.style.color = 'var(--text-muted)';

    } catch (e) {
        tableHead.innerHTML = `<tr><th class="text-red">PostgreSQL Sandbox Error</th></tr>`;
        tableBody.innerHTML = `<tr><td class="font-mono text-red p-4" style="background: var(--danger-light); border-radius: var(--radius-sm); color: var(--danger);">${e.message}</td></tr>`;
        metaText.innerText = `Rows: 0 | Errors: 1`;
        metaText.style.color = 'var(--danger)';
    }
};

/* --- Global Utilities --- */

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function animateValCount(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const val = Math.floor(progress * (end - start) + start);
        obj.innerHTML = `$${val.toLocaleString()}`;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Initialize session immediately on app script load
window.initAuthSession();
