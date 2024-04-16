const std = @import("std");

pub fn Relu(size: usize) type {
    return struct {
        last_inputs: []f64 = undefined,

        const Self = @This();

        fn activation(in: f64) f64 {
            if (in < 0) {
                return 0.01 * in;
            } else {
                return in;
            }
        }

        pub fn forward(self: *Self, inputs: []f64) []f64 {
            std.debug.assert(inputs.len == size);
            var outputs: [size]f64 = [_]f64{0} ** size;

            var i: usize = 0;
            while (i < inputs.len) : (i += 1) {
                if (inputs[i] < 0) {
                    outputs[i] = 0.01 * inputs[i];
                } else {
                    outputs[i] = inputs[i];
                }
            }
            self.last_inputs = inputs;
            return &outputs;
        }

        pub fn backwards(self: *Self, grads: []f64) []f64 {
            std.debug.assert(grads.len == size);
            var outputs: [size]f64 = [_]f64{0} ** size;
            var i: usize = 0;
            while (i < self.last_inputs.len) : (i += 1) {
                if (self.last_inputs[i] < 0) {
                    outputs[i] = 0.01 * grads[i];
                } else {
                    outputs[i] = grads[i];
                }
            }
            return &outputs;
        }
    };
}
