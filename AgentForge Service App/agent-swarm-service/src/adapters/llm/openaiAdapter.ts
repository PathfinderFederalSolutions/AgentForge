export class OpenAIAdapter {
    private apiKey: string;
    private endpoint: string;

    constructor(apiKey: string) {
        this.apiKey = apiKey;
        this.endpoint = 'https://api.openai.com/v1/engines/davinci-codex/completions';
    }

    async generateResponse(prompt: string): Promise<string> {
        const response = await fetch(this.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`,
            },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: 150,
                n: 1,
                stop: null,
                temperature: 0.7,
            }),
        });

        if (!response.ok) {
            throw new Error(`OpenAI API error: ${response.statusText}`);
        }

        const data = await response.json();
        return data.choices[0].text.trim();
    }
}