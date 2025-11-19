/**
 * AgentForge Batch Upload Solution
 * Handles massive file uploads (2000+ files) by splitting into manageable batches
 */

class AgentForgeBatchUploader {
    constructor(apiBaseUrl = 'http://localhost:8000') {
        this.apiBaseUrl = apiBaseUrl;
        this.batchSize = 50; // Files per batch
        this.maxConcurrentBatches = 3; // Concurrent uploads
    }

    /**
     * Upload massive file sets in batches
     * @param {FileList|File[]} files - Files to upload
     * @param {Function} onProgress - Progress callback (current, total, percentage)
     * @param {Function} onBatchComplete - Batch completion callback
     * @returns {Promise<Object>} Upload results
     */
    async uploadMassiveFileSet(files, onProgress = null, onBatchComplete = null) {
        const fileArray = Array.from(files);
        const totalFiles = fileArray.length;
        const totalBatches = Math.ceil(totalFiles / this.batchSize);
        
        console.log(`🚀 Starting massive upload: ${totalFiles} files in ${totalBatches} batches`);
        
        const results = {
            totalFiles,
            totalBatches,
            completedBatches: 0,
            successfulFiles: 0,
            failedFiles: 0,
            batchResults: [],
            startTime: Date.now()
        };

        // Process batches with concurrency control
        const batchPromises = [];
        for (let i = 0; i < totalBatches; i += this.maxConcurrentBatches) {
            const concurrentBatches = [];
            
            for (let j = 0; j < this.maxConcurrentBatches && (i + j) < totalBatches; j++) {
                const batchIndex = i + j;
                const startIdx = batchIndex * this.batchSize;
                const endIdx = Math.min(startIdx + this.batchSize, totalFiles);
                const batchFiles = fileArray.slice(startIdx, endIdx);
                
                concurrentBatches.push(
                    this.uploadBatch(batchFiles, batchIndex, totalBatches)
                        .then(result => {
                            results.completedBatches++;
                            results.successfulFiles += result.files_processed || 0;
                            results.batchResults.push(result);
                            
                            // Progress callback
                            if (onProgress) {
                                const percentage = (results.completedBatches / totalBatches) * 100;
                                onProgress(results.completedBatches, totalBatches, percentage);
                            }
                            
                            // Batch complete callback
                            if (onBatchComplete) {
                                onBatchComplete(result, batchIndex, totalBatches);
                            }
                            
                            console.log(`✅ Batch ${batchIndex + 1}/${totalBatches} complete`);
                            return result;
                        })
                        .catch(error => {
                            console.error(`❌ Batch ${batchIndex + 1} failed:`, error);
                            results.failedFiles += batchFiles.length;
                            return { error: error.message, batchIndex };
                        })
                );
            }
            
            // Wait for current batch group to complete before starting next
            await Promise.all(concurrentBatches);
        }

        results.endTime = Date.now();
        results.totalTime = results.endTime - results.startTime;
        
        console.log(`🎉 Massive upload complete: ${results.successfulFiles}/${totalFiles} files uploaded in ${results.totalTime/1000}s`);
        
        return results;
    }

    /**
     * Upload a single batch of files
     * @param {File[]} files - Files in this batch
     * @param {number} batchIndex - Batch number
     * @param {number} totalBatches - Total number of batches
     * @returns {Promise<Object>} Batch result
     */
    async uploadBatch(files, batchIndex, totalBatches) {
        const formData = new FormData();
        
        // Add files to form data
        files.forEach(file => {
            formData.append('files', file);
        });
        
        // Add batch metadata
        formData.append('batch_id', `massive_upload_${Date.now()}`);
        formData.append('batch_index', batchIndex.toString());
        
        console.log(`📦 Uploading batch ${batchIndex + 1}/${totalBatches} (${files.length} files)`);
        
        const response = await fetch(`${this.apiBaseUrl}/v1/io/upload-batch`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Batch upload failed: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    }

    /**
     * Get upload system information
     * @returns {Promise<Object>} Upload system info
     */
    async getUploadInfo() {
        const response = await fetch(`${this.apiBaseUrl}/v1/io/upload-info`);
        return await response.json();
    }
}

// Usage Example:
/*
const uploader = new AgentForgeBatchUploader();

// Handle file input change
document.getElementById('fileInput').addEventListener('change', async (event) => {
    const files = event.target.files;
    
    if (files.length > 500) {
        console.log('Large upload detected, using batch uploader...');
        
        const results = await uploader.uploadMassiveFileSet(
            files,
            // Progress callback
            (current, total, percentage) => {
                console.log(`Progress: ${current}/${total} batches (${percentage.toFixed(1)}%)`);
                // Update progress bar here
            },
            // Batch complete callback
            (result, batchIndex, totalBatches) => {
                console.log(`Batch ${batchIndex + 1}/${totalBatches} completed`);
                // Update UI here
            }
        );
        
        console.log('Upload complete:', results);
    } else {
        // Use regular upload for smaller file sets
        // ... regular upload code ...
    }
});
*/

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AgentForgeBatchUploader;
}

// Make available globally
if (typeof window !== 'undefined') {
    window.AgentForgeBatchUploader = AgentForgeBatchUploader;
}



