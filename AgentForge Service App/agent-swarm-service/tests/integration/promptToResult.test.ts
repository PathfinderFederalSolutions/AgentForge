import { promptToResult } from '../../src/workflows/promptToResult';
import { IngestionService } from '../../src/services/ingestionService';
import { ProcessingService } from '../../src/services/processingService';
import { OutputService } from '../../src/services/outputService';

describe('Integration Tests for promptToResult Workflow', () => {
    let ingestionService: IngestionService;
    let processingService: ProcessingService;
    let outputService: OutputService;

    beforeAll(() => {
        ingestionService = new IngestionService();
        processingService = new ProcessingService();
        outputService = new OutputService();
    });

    test('should process user prompt and return a result', async () => {
        const userPrompt = 'What is the capital of France?';
        const expectedResult = 'The capital of France is Paris.';

        const result = await promptToResult(userPrompt, ingestionService, processingService, outputService);

        expect(result).toBe(expectedResult);
    });

    // Additional integration tests can be added here
});