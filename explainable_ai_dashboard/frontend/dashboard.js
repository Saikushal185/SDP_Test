/**
 * Dashboard JavaScript
 * Handles data fetching, rendering, and interactions
 */

const API_BASE_URL = 'http://localhost:5000/api';

// Update timestamp
function updateTimestamp() {
    const now = new Date();
    const timestamp = now.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    document.getElementById('timestamp').textContent = timestamp;
}

// Update gauge needle based on severity
function updateGaugeNeedle(severity) {
    const needle = document.getElementById('gaugeNeedle');
    if (!needle) return;

    // Convert severity (0-1) to angle (-90 to +90 degrees)
    const angle = (severity * 180) - 90;

    // Calculate needle endpoint
    const centerX = 100;
    const centerY = 100;
    const length = 70;
    const radians = (angle * Math.PI) / 180;
    const endX = centerX + length * Math.cos(radians);
    const endY = centerY + length * Math.sin(radians);

    needle.setAttribute('x2', endX);
    needle.setAttribute('y2', endY);
}

// Load patient data
async function loadPatientData(patientId) {
    try {
        // Add loading state
        document.querySelectorAll('.card').forEach(card => {
            card.classList.add('loading');
        });

        const response = await fetch(`${API_BASE_URL}/patient/${patientId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch patient data');
        }

        const data = await response.json();
        renderPatientData(data);

        // Remove loading state
        document.querySelectorAll('.card').forEach(card => {
            card.classList.remove('loading');
        });

    } catch (error) {
        console.error('Error loading patient data:', error);
        alert('Failed to load patient data. Make sure the API server is running.');
    }
}

// Render patient data to dashboard
function renderPatientData(data) {
    // Primary Diagnosis
    const statusBadge = document.getElementById('statusBadge');
    const probability = Math.round(data.primary_diagnosis.probability * 100) / 100;

    document.getElementById('diagnosisCondition').textContent = data.primary_diagnosis.condition;
    document.getElementById('probability').textContent = probability.toFixed(2);
    document.getElementById('affectedRegion').textContent =
        `${data.primary_diagnosis.primary_affected_region}, ${data.primary_diagnosis.region_severity}`;

    statusBadge.textContent = data.primary_diagnosis.status;
    statusBadge.className = 'status-badge';
    if (data.primary_diagnosis.status === 'Positive') {
        statusBadge.style.background = '#10B981';
    } else {
        statusBadge.style.background = '#6B7280';
    }

    // Decision Basis
    const basisContainer = document.getElementById('decisionBasis');
    basisContainer.innerHTML = data.primary_diagnosis.decision_basis
        .map(item => `<div class="basis-item">✓ ${item}</div>`)
        .join('');

    // Non-Motor Risks
    // Cognitive
    document.getElementById('cognitiveScore').textContent =
        data.non_motor_risks.cognitive_decline.risk_score.toFixed(2);
    document.getElementById('cognitiveAssociationBin').textContent =
        data.non_motor_risks.cognitive_decline.associated_bin;

    const cognitiveIndicators = document.getElementById('cognitiveIndicators');
    cognitiveIndicators.innerHTML = data.non_motor_risks.cognitive_decline.indicators
        .map(ind => `<div class="indicator-item">✓ ${ind}</div>`)
        .join('');

    // Depression
    document.getElementById('depressionScore').textContent =
        data.non_motor_risks.depression.risk_score.toFixed(2);
    document.getElementById('depressionRisk').textContent =
        data.non_motor_risks.depression.associated_risk;

    const depressionIndicators = document.getElementById('depressionIndicators');
    depressionIndicators.innerHTML = data.non_motor_risks.depression.indicators
        .map(ind => `<div class="indicator-item">✓ ${ind}</div>`)
        .join('');

    // Dysphagia
    document.getElementById('dysphagiaScore').textContent =
        data.non_motor_risks.dysphagia.risk_score.toFixed(2);
    document.getElementById('speechProxy').textContent =
        data.non_motor_risks.dysphagia.speech_proxy;

    const dysphagiaIndicators = document.getElementById('dysphagiaIndicators');
    dysphagiaIndicators.innerHTML = data.non_motor_risks.dysphagia.indicators
        .map(ind => `<div class="indicator-item">✓ ${ind}</div>`)
        .join('');

    // Motor Speech Assessment
    const impairmentLevel = document.getElementById('impairmentLevel');
    impairmentLevel.textContent = data.motor_speech_assessment.impairment_level;
    impairmentLevel.className = `badge badge-${data.motor_speech_assessment.impairment_level.toLowerCase()}`;

    document.getElementById('severityScore').textContent =
        data.motor_speech_assessment.severity_score.toFixed(2);

    // Speech features
    const speechFeatures = document.getElementById('speechFeatures');
    if (data.motor_speech_assessment.jitter && data.motor_speech_assessment.jitter.indicators) {
        speechFeatures.innerHTML = data.motor_speech_assessment.jitter.indicators
            .map(feat => `<div class="feature-item">• ${feat}</div>`)
            .join('');
    }

    // Voice instability
    const voiceInstability = document.getElementById('voiceInstability');
    voiceInstability.textContent = data.motor_speech_assessment.voice_instability.level;
    voiceInstability.className = `badge badge-${data.motor_speech_assessment.voice_instability.level.toLowerCase()}`;

    // Motor features
    const motorFeatures = document.getElementById('motorFeatures');
    motorFeatures.innerHTML = data.motor_speech_assessment.motor_features
        .map(feat => `<span>✓ ${feat}</span>`)
        .join('');

    // Prosodic features
    const prosodicFeatures = document.getElementById('prosodicFeatures');
    prosodicFeatures.innerHTML = data.motor_speech_assessment.prosodic_features
        .map(feat => `<span>✓ ${feat}</span>`)
        .join('');

    // Overall Risk Assessment
    document.getElementById('severityIndex').textContent =
        data.overall_risk_assessment.severity_index.toFixed(2);
    document.getElementById('diseaseStage').textContent =
        data.overall_risk_assessment.disease_stage;
    document.getElementById('progressionRisk').textContent =
        data.overall_risk_assessment.progression_risk;

    // Update gauge
    updateGaugeNeedle(data.overall_risk_assessment.severity_index);

    // Recommendations
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = data.recommendations
        .map(rec => `<div class="recommendation-item">• ${rec}</div>`)
        .join('');
}

// Load patient list
async function loadPatientList() {
    try {
        const response = await fetch(`${API_BASE_URL}/patients/list`);
        if (!response.ok) {
            throw new Error('Failed to fetch patient list');
        }

        const data = await response.json();
        const select = document.getElementById('patientSelect');

        select.innerHTML = data.patients
            .map(patient => `<option value="${patient.patient_id}">${patient.label}</option>`)
            .join('');

    } catch (error) {
        console.error('Error loading patient list:', error);
        // Keep default options if API call fails
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    updateTimestamp();
    setInterval(updateTimestamp, 60000); // Update every minute

    // Patient selection
    const patientSelect = document.getElementById('patientSelect');
    patientSelect.addEventListener('change', (e) => {
        loadPatientData(e.target.value);
    });

    // Initial load
    loadPatientList().then(() => {
        loadPatientData(0); // Load first patient
    });
});

// Handle API connection errors gracefully
window.addEventListener('unhandledrejection', (event) => {
    if (event.reason && event.reason.message && event.reason.message.includes('fetch')) {
        console.warn('API connection issue - using fallback data');
        event.preventDefault();
    }
});
