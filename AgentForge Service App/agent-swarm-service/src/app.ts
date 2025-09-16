import express from 'express';
import { json } from 'body-parser';
import { IngestionService } from './services/ingestionService';
import { ProcessingService } from './services/processingService';
import { OutputService } from './services/outputService';
import { Coordinator } from './agents/coordinator';
import { Orchestrator } from './orchestrator/orchestrator';
import { logInfo, logError } from './utils/logger';

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(json());
app.use((req, res, next) => {
    logInfo(`Received request: ${req.method} ${req.url}`);
    next();
});

// Services and orchestrator setup
const ingestionService = new IngestionService();
const processingService = new ProcessingService();
const outputService = new OutputService();
const coordinator = new Coordinator([ingestionService, processingService, outputService]);
const orchestrator = new Orchestrator([coordinator]);

// Routes
app.post('/process', async (req, res) => {
    try {
        const result = await orchestrator.processRequest(req.body);
        res.status(200).json(result);
    } catch (error: any) {
        logError(`Error processing request: ${error.message ?? error}`);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// Start the server
app.listen(port, () => {
    logInfo(`Agent swarm service listening on port ${port}`);
});

export = app;