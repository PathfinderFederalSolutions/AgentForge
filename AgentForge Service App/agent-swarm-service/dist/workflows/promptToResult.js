"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.promptToResult = void 0;
const ingestionService_1 = require("../services/ingestionService");
const ingestionServiceInstance = ingestionService_1.ingestionService;
const processingServiceInstance = ingestionService_1.processingService;
const outputServiceInstance = ingestionService_1.outputService;
const promptToResult = async (userPrompt) => {
    // Initialize services and agents
    const ingestionService = ingestionServiceInstance;
    const processingService = processingServiceInstance;
    const outputService = outputServiceInstance;
    try {
        // Ingest data
        await ingestionService.ingestData(userPrompt);
        const isValid = await ingestionService.validateData(userPrompt);
        if (!isValid) {
            throw new Error('Invalid data');
        }
        const ingestedData = await ingestionService.transformData(userPrompt);
        // Process data
        const processedData = processingService.processData(ingestedData);
        const interpretedResults = processingService.interpretResults(processedData);
        // Format and deliver output
        const result = outputService.formatResult(interpretedResults);
        outputService.deliverResult(result, 'default-destination');
        return result;
    }
    catch (error) {
        // Handle errors appropriately
        console.error("Error in promptToResult workflow:", error);
        throw new Error("Failed to process the prompt.");
    }
};
exports.promptToResult = promptToResult;
