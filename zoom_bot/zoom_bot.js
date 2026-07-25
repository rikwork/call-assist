const { chromium } = require('playwright');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const path = require('path');

// Load environment variables from parent directory
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });
const args = process.argv.slice(2);
const params = {};
for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
        const key = args[i].slice(2);
        const val = args[i + 1];
        params[key] = val;
        i++;
    }
}

let zoomUrl = params.url;
if (zoomUrl && zoomUrl.includes('/j/')) {
    zoomUrl = zoomUrl.replace('/j/', '/wc/join/');
}
const sessionId = params.session || 'default_session';
const backendHost = params.backend || 'localhost:8000';

if (!zoomUrl) {
    console.error('Error: Please provide --url <Zoom Invite URL>');
    process.exit(1);
}

// Format WS URI
const wsProtocol = backendHost.includes('localhost') ? 'ws' : 'wss';
const wsUri = `${wsProtocol}://${backendHost}/audio/${sessionId}?source=bot`;

console.log(`[Zoom Bot] Target Session: ${sessionId}`);
console.log(`[Zoom Bot] Target Zoom URL: ${zoomUrl}`);
console.log(`[Zoom Bot] Target WebSocket: ${wsUri}`);

async function run() {
    // 1. Connect to FastAPI WebSocket Server
    console.log('[Zoom Bot] Connecting to backend WebSocket...');
    const ws = new WebSocket(wsUri);
    
    let parecProcess = null;

    ws.on('open', () => {
        console.log('[Zoom Bot] Connected to backend WebSocket successfully.');
    });

    ws.on('close', () => {
        console.log('[Zoom Bot] Backend WebSocket closed. Terminating bot...');
        if (parecProcess) parecProcess.kill();
        process.exit(0);
    });

    ws.on('error', (err) => {
        console.error('[Zoom Bot] WebSocket error:', err);
    });

    // 2. Launch Chromium (Playwright)
    // We run in headful mode inside Xvfb so Zoom WebRTC works perfectly
    console.log('[Zoom Bot] Launching Chromium browser...');
    const browser = await chromium.launch({
        headless: false, 
        ignoreDefaultArgs: ['--enable-automation'],
        args: [
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream',
            '--autoplay-policy=no-user-gesture-required',
            '--disable-blink-features=AutomationControlled'
        ]
    });

    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        permissions: ['microphone', 'camera']
    });

    const page = await context.newPage();
    
    // Hide webdriver flag to bypass basic bot detectors
    await page.addInitScript(() => {
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    });
    
    try {
        console.log(`[Zoom Bot] Navigating to Zoom URL...`);
        await page.goto(zoomUrl, { waitUntil: 'domcontentloaded', timeout: 25000 });

        // Bypassing landing page to open web client
        // Look for "Join from your browser" link
        console.log('[Zoom Bot] Checking for Web Client join link...');
        try {
            await page.waitForSelector('a[href*="/wc/join"]', { timeout: 5000 });
            await page.click('a[href*="/wc/join"]');
            console.log('[Zoom Bot] Clicked "Join from your browser".');
        } catch (e) {
            console.log('[Zoom Bot] Direct web client link not found, attempting to bypass Zoom launcher...');
            // Sometimes it's a direct web client URL, so we continue
        }

        // Enter Name and Join
        console.log('[Zoom Bot] Waiting for Name input...');
        const nameInputSelector = 'input[name="name"], #inputname, input[type="text"]';
        await page.waitForSelector(nameInputSelector, { timeout: 15000 });
        await page.fill(nameInputSelector, 'Никита');
        
        console.log('[Zoom Bot] Clicking Join Button...');
        const joinBtnSelector = 'button.wc-btn-join, #joinBtn, button:has-text("Join"), button:has-text("Войти"), button:has-text("Присоединиться"), button.preview-join-button';
        await page.waitForSelector(joinBtnSelector, { timeout: 5000 });
        await page.click(joinBtnSelector);

        // Bypassing passcode if asked (URL usually contains password, but just in case)
        try {
            const passcodeSelector = 'input[name="passcode"], #inputpasscode';
            if (await page.$(passcodeSelector)) {
                console.log('[Zoom Bot] Meeting passcode required. Waiting for manual entry or bypass...');
            }
        } catch (e) {}

        // Connect audio
        console.log('[Zoom Bot] Waiting for Join Audio dialog...');
        const joinAudioBtnSelector = 'button:has-text("Join Audio"), button:has-text("Join Computer Audio"), button.join-audio-btn';
        await page.waitForSelector(joinAudioBtnSelector, { timeout: 20000 });
        await page.click(joinAudioBtnSelector);
        console.log('[Zoom Bot] Clicked "Join Computer Audio" successfully.');

        // 3. Start Recording Audio from Virtual Sink
        console.log('[Zoom Bot] Initializing PulseAudio recording...');
        
        // Spawn parec to record 16kHz Mono 16-bit signed PCM audio
        parecProcess = spawn('parec', [
            '--device=VirtualSink.monitor',
            '--format=s16le',
            '--rate=16000',
            '--channels=1',
            '--client-name=ZoomBotRecorder'
        ]);

        // Deepgram WebSockets integration for real-time speech-to-text
        let dgWs = null;
        const deepgramApiKey = process.env.DEEPGRAM_API_KEY;
        
        if (deepgramApiKey) {
            console.log('[Zoom Bot] Deepgram API Key found. Initializing streaming STT...');
            // Connect to Deepgram live transcription WebSocket
            dgWs = new WebSocket('wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&diarize=true&language=ru', {
                headers: {
                    Authorization: `Token ${deepgramApiKey}`
                }
            });

            dgWs.on('open', () => {
                console.log('[Zoom Bot] Connected to Deepgram STT stream successfully.');
            });

            dgWs.on('message', (message) => {
                try {
                    const data = JSON.parse(message);
                    const transcript = data.channel?.alternatives?.[0]?.transcript;
                    if (transcript && transcript.trim().length > 0) {
                        // Diarization: get speaker ID (usually 0, 1, 2...)
                        // Deepgram tags each word with speaker index, we check the first word
                        const speakerId = data.channel?.alternatives?.[0]?.words?.[0]?.speaker ?? 1;
                        
                        // We map Speaker 0 to Representative and Speaker 1 (or others) to Customer
                        // Generally, the representative initiates the call, but we can dynamically map.
                        // For simplicity: Speaker 0 is Representative (Менеджер), Speaker 1+ is Customer (Клиент)
                        const speakerRole = (speakerId === 0) ? 'Representative' : 'Customer';
                        
                        console.log(`[Zoom Bot] [STT Output] Speaker ${speakerId} (${speakerRole}): ${transcript}`);
                        
                        // Send text transcript packet back to FastAPI backend
                        if (ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                type: 'transcript',
                                speaker: speakerRole,
                                text: transcript
                            }));
                        }
                    }
                } catch (e) {
                    console.error('[Zoom Bot] Error parsing Deepgram message:', e);
                }
            });

            dgWs.on('error', (err) => {
                console.error('[Zoom Bot] Deepgram WebSocket error:', err);
            });

            dgWs.on('close', () => {
                console.log('[Zoom Bot] Deepgram WebSocket connection closed.');
            });
        } else {
            console.log('[Zoom Bot] WARNING: No DEEPGRAM_API_KEY found in .env. Streaming raw audio only (will require local STT).');
        }

        parecProcess.stdout.on('data', (chunk) => {
            // 1. Send raw audio to our FastAPI server for VAD silence detection
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(chunk);
            }
            // 2. Stream raw audio to Deepgram for live transcription
            if (dgWs && dgWs.readyState === WebSocket.OPEN) {
                dgWs.send(chunk);
            }
        });

        parecProcess.stderr.on('data', (data) => {
            console.log(`[parec stderr]: ${data}`);
        });

        parecProcess.on('close', (code) => {
            console.log(`[Zoom Bot] parec process exited with code ${code}`);
            if (dgWs && dgWs.readyState === WebSocket.OPEN) {
                dgWs.close();
            }
        });

        console.log('[Zoom Bot] Audio capture started. Streaming to backend...');
        
        // Keep running until manually stopped or meeting disconnected
        while (true) {
            await page.waitForTimeout(5000);
            if (page.isClosed()) break;
        }

    } catch (error) {
        console.error('[Zoom Bot] Error during automation:', error);
        try {
            const screenshotPath = path.resolve(__dirname, 'bot_error.png');
            await page.screenshot({ path: screenshotPath });
            console.log(`[Zoom Bot] Saved error screenshot to ${screenshotPath}`);
        } catch (e) {
            console.error('[Zoom Bot] Failed to capture error screenshot:', e);
        }
    } finally {
        if (parecProcess) parecProcess.kill();
        await browser.close();
        process.exit(0);
    }
}

run();
