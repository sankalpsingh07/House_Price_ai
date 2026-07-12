-- RealEstateAI Portfolio Planner & Price Predictor
-- PostgreSQL Database Schema

-- Enable UUID extension for cryptographic keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. USERS TABLE
-- Stores system user accounts
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. PORTFOLIOS TABLE
-- Groups individual properties together under an investment package
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. PROPERTIES TABLE
-- Stores property features (inputs) and the corresponding ML predicted price (outputs)
CREATE TABLE IF NOT EXISTS properties (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE SET NULL,
    address VARCHAR(255) NOT NULL,
    median_income NUMERIC(10, 4) NOT NULL, -- in tens of thousands (e.g. 8.32 = $83,200)
    house_age NUMERIC(5, 1) NOT NULL, -- years
    ave_rooms NUMERIC(6, 2) NOT NULL, -- average rooms
    ave_bedrooms NUMERIC(6, 2) NOT NULL, -- average bedrooms
    population INT NOT NULL, -- total population in block
    ave_occupancy NUMERIC(6, 2) NOT NULL, -- average occupants
    latitude NUMERIC(6, 3) NOT NULL, -- spatial location
    longitude NUMERIC(6, 3) NOT NULL, -- spatial location
    actual_value NUMERIC(12, 2), -- actual price in USD (if available)
    predicted_value NUMERIC(12, 2) NOT NULL, -- calculated price in USD (OLS model output)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. PREDICTION HISTORY TABLE
-- Log table acting as an audit history for property re-valuations over time
CREATE TABLE IF NOT EXISTS prediction_history (
    id SERIAL PRIMARY KEY,
    property_id UUID NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL DEFAULT 'linear-regression-ols-v1',
    predicted_value NUMERIC(12, 2) NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index optimization for faster queries
CREATE INDEX IF NOT EXISTS idx_properties_portfolio ON properties(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolios_user ON portfolios(user_id);
CREATE INDEX IF NOT EXISTS idx_prediction_history_property ON prediction_history(property_id);

-- Insert sample demonstration records
INSERT INTO users (id, email, full_name) VALUES
('d3b07384-d113-4c4e-9c8e-5b1234567890', 'investor@realestateai.com', 'Alex Mercer')
ON CONFLICT (email) DO NOTHING;

INSERT INTO portfolios (id, user_id, name, description) VALUES
('b9a2e38c-8f4f-4d6f-9988-776655443322', 'd3b07384-d113-4c4e-9c8e-5b1234567890', 'Silicon Valley Premium', 'Premium properties located near technology centers in Santa Clara and SF.')
ON CONFLICT DO NOTHING;

INSERT INTO properties (id, portfolio_id, address, median_income, house_age, ave_rooms, ave_bedrooms, population, ave_occupancy, latitude, longitude, actual_value, predicted_value) VALUES
('f47ac10b-58cc-4372-a567-0e02b2c3d4e5', 'b9a2e38c-8f4f-4d6f-9988-776655443322', '345 University Ave, Palo Alto', 8.3252, 41, 6.98, 1.02, 322, 2.55, 37.88, -122.23, 452600.00, 412500.00),
('a82bc10b-58cc-4372-a567-0e02b2c3d4f6', 'b9a2e38c-8f4f-4d6f-9988-776655443322', '120 Hawthorne St, Palo Alto', 7.2574, 52, 8.28, 1.07, 496, 2.80, 37.85, -122.24, 352100.00, 348200.00)
ON CONFLICT DO NOTHING;
