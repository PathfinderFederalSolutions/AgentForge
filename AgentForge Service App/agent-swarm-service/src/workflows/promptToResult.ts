export const promptToResult = async (userPrompt: string): Promise<any> => {
    // Initialize services and agents
    const ingestionService = new IngestionService();
    const processingService = new ProcessingService();
    const outputService = new OutputService();

    try {
        // Step 1: Ingest data based on the user prompt
        const ingestedData = await ingestionService.ingest(userPrompt);

        // Step 2: Process the ingested data
        const processedData = await processingService.process(ingestedData);

        // Step 3: Generate the final result
        const result = await outputService.formatOutput(processedData);

        return result;
    } catch (error) {
        // Handle errors appropriately
        console.error("Error in promptToResult workflow:", error);
        throw new Error("Failed to process the prompt.");
    }
};