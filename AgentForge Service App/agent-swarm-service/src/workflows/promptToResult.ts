import { ingestionService, processingService, outputService } from '../services/ingestionService';

const ingestionServiceInstance = ingestionService;
const processingServiceInstance = processingService;
const outputServiceInstance = outputService;

export const promptToResult = async (userPrompt: string): Promise<any> => {
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
    } catch (error) {
        // Handle errors appropriately
        console.error("Error in promptToResult workflow:", error);
        throw new Error("Failed to process the prompt.");
    }
};