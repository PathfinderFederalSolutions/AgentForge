"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OpenAIAdapter = void 0;
class OpenAIAdapter {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.endpoint = 'https://api.openai.com/v1/engines/davinci-codex/completions';
    }
    async generateResponse(prompt) {
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
exports.OpenAIAdapter = OpenAIAdapter;
