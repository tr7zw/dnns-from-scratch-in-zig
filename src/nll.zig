const std = @import("std");

const GiveLoss = true;

pub fn NLL(
    comptime inputSize: usize,
    //comptime batchSize: usize,
) type {
    return struct {
        loss: []f64, // = [1]f64{0} ** (batchSize);
        input_grads: []f64, // = [1]f64{0} ** (batchSize * inputSize);
        batchSize: usize,

        const Self = @This();

        pub fn init(
            alloc: std.mem.Allocator,
            batchSize: usize,
        ) !Self {
            return Self{
                .loss = try alloc.alloc(f64, batchSize),
                .input_grads = try alloc.alloc(f64, batchSize * inputSize),
                .batchSize = batchSize,
            };
        }
        pub fn nll(self: *Self, inputs: []f64, targets: []u8) !void {
            var b: usize = 0;
            while (b < self.batchSize) : (b += 1) {
                var sum: f64 = 0;
                var i: usize = 0;
                while (i < inputSize) : (i += 1) {
                    sum += std.math.exp(inputs[b * inputSize + i]);
                    if (sum == std.math.inf(f64)) {
                        std.debug.print("output with inf:\n {any},\n", .{
                            inputs[b * inputSize + i],
                        });
                        return error.divisionbyinf;
                    }
                }
                const s = sum;
                //std.math.sign(sum + 0.0000001) *
                sum = (std.math.sign(sum) + 0.000000001) * @max(0.0000001, @abs(sum));
                if (sum >= std.math.inf(f64)) return error.divisionbyinf;
                if (std.math.sign(sum) == 0) {
                    std.debug.print("sum:{}\n", .{s});
                    return error.sumisnan;
                }
                if (sum == 0) {
                    return error.divisionbyzero;
                }

                if (GiveLoss) {
                    self.loss[b] = -1 * @log(std.math.exp(inputs[b * inputSize + targets[b]]) / sum);
                }
                i = 0;
                while (i < inputSize) : (i += 1) {
                    self.input_grads[b * inputSize + i] = std.math.exp(inputs[b * inputSize + i]) / sum;
                    if (i == targets[b]) {
                        self.input_grads[b * inputSize + i] -= 1;
                    }
                }
            }
        }
    };
}
