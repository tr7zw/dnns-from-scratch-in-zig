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

            var outputs: [size]f64 = undefined;

            var i: usize = 0;
            while (i < inputs.len) : (i += 1) {
                outputs[i] = activation(inputs[i]);
            }
            self.last_inputs = inputs;
            return &outputs;
        }

        pub fn backwards(self: *Self, grads: []f64) []f64 {
            std.debug.assert(grads.len == size);
            var outputs: [size]f64 = undefined;
            var i: usize = 0;
            while (i < self.last_inputs.len) : (i += 1) {
                outputs[i] = activation(grads[i]);
            }
            return &outputs;
        }
    };
}
