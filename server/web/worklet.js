// Runs on the browser's audio thread: hands the page ~100 ms blocks of each input channel,
// the microphone alone in dictation, the microphone and the shared tab's audio in a meeting.
class Capture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.channels = (options.processorOptions && options.processorOptions.channels) || 1;
    this.block = Math.round(sampleRate / 10);
    this.fresh();
  }

  fresh() {
    this.buffers = Array.from({ length: this.channels }, () => new Float32Array(this.block));
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    const length = input && input.length ? input[0].length : 0;
    let offset = 0;
    while (offset < length) {
      const n = Math.min(length - offset, this.block - this.filled);
      for (let c = 0; c < this.channels; c++) {
        if (input[c]) this.buffers[c].set(input[c].subarray(offset, offset + n), this.filled); // else stays silent
      }
      this.filled += n;
      offset += n;
      if (this.filled === this.block) {
        this.port.postMessage(this.buffers);
        this.fresh();
      }
    }
    return true;
  }
}

registerProcessor("capture", Capture);
