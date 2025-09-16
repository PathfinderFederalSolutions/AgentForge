"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const express_1 = __importDefault(require("express"));
const body_parser_1 = require("body-parser");
const ingestionService_1 = require("./services/ingestionService");
const processingService_1 = require("./services/processingService");
const outputService_1 = require("./services/outputService");
const coordinator_1 = require("./agents/coordinator");
const orchestrator_1 = require("./orchestrator/orchestrator");
const logger_1 = require("./utils/logger");
const app = (0, express_1.default)();
const port = process.env.PORT || 3000;
// Middleware
app.use((0, body_parser_1.json)());
app.use((req, res, next) => {
    (0, logger_1.logInfo)(`Received request: ${req.method} ${req.url}`);
    next();
});
// Services and orchestrator setup
const ingestionService = new ingestionService_1.IngestionService();
const processingService = new processingService_1.ProcessingService();
const outputService = new outputService_1.OutputService();
const coordinator = new coordinator_1.Coordinator([ingestionService, processingService, outputService]);
const orchestrator = new orchestrator_1.Orchestrator([coordinator]);
// Routes
app.post('/process', async (req, res) => {
    try {
        const result = await orchestrator.processRequest(req.body);
        res.status(200).json(result);
    }
    catch (error) {
        (0, logger_1.logError)(`Error processing request: ${error.message ?? error}`);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
// Start the server
app.listen(port, () => {
    (0, logger_1.logInfo)(`Agent swarm service listening on port ${port}`);
});
module.exports = app;
