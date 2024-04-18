const layer = @import("layer.zig");
const nll = @import("nll.zig");
const mnist = @import("mnist.zig");
const relu = @import("relu.zig");

const std = @import("std");

const INPUT_SIZE: u32 = 784;
const OUTPUT_SIZE: u32 = 10;
const LAYER_SIZE: u32 = 100;

const BATCH_SIZE: u32 = 100;

const EPOCHS: u32 = 8;

const l1 = layer.Layer(INPUT_SIZE, LAYER_SIZE, BATCH_SIZE);
const Relu1 = relu.Relu(LAYER_SIZE * BATCH_SIZE);
const l2 = layer.Layer(LAYER_SIZE, OUTPUT_SIZE, BATCH_SIZE);
const Loss = nll.NLL(OUTPUT_SIZE, BATCH_SIZE);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // Get MNIST data
    const mnist_data = try mnist.readMnist(allocator);
    defer mnist_data.deinit(allocator);
    // Prep NN
    var layer1: l1 = l1.init();
    var relu1: Relu1 = Relu1.init();
    var layer2: l2 = l2.init();
    var loss: Loss = .{
        .input_grads = [1]f64{0} ** (BATCH_SIZE * OUTPUT_SIZE),
        .loss = [1]f64{0} ** (BATCH_SIZE),
    };

    var t = std.time.milliTimestamp();
    std.debug.print("Training... \n", .{});
    // Do training
    var e: usize = 0;
    while (e < EPOCHS) : (e += 1) {
        // Do training
        var i: usize = 0;
        while (i < 60000 / BATCH_SIZE) : (i += 1) {

            // Prep inputs and targets
            const inputs = mnist_data.train_images[i * INPUT_SIZE * BATCH_SIZE .. (i + 1) * INPUT_SIZE * BATCH_SIZE];
            const targets = mnist_data.train_labels[i * BATCH_SIZE .. (i + 1) * BATCH_SIZE];

            // Go forward and get loss
            const outputs1 = layer1.forward(inputs);
            const outputs2 = relu1.forward(outputs1.outputs);
            const outputs3 = layer2.forward(&outputs2.fwd_out);
            loss = loss.nll(&outputs3.outputs, targets).*;
            if (i % (10000 / BATCH_SIZE) == 0) {
                const ct = std.time.milliTimestamp();
                std.debug.print("batch number: {}, time delta: {}ms\n", .{ i, ct - t });
                std.debug.print("average loss for batch: {any}, avg gradient {any}\n", .{ averageArray(&loss.loss), averageArray(&loss.input_grads) });
                t = ct;
            }
            // Update network
            const grads1 = layer2.backwards(&loss.input_grads);
            const grads2 = relu1.backwards(&grads1.input_grads);
            const grads3 = layer1.backwards(&grads2.bkw_out);
            layer1.applyGradients(&grads3.weight_grads);
            layer2.applyGradients(&grads1.weight_grads);
        }

        // Do validation
        i = 0;
        var correct: f64 = 0;
        var b: usize = 0;
        while (b < 10000 / BATCH_SIZE) : (b += 1) {
            const inputs = mnist_data.test_images[b * INPUT_SIZE * BATCH_SIZE .. (b + 1) * INPUT_SIZE * BATCH_SIZE];

            const outputs1 = layer1.forward(inputs);
            const outputs2 = relu1.forward(outputs1.outputs);
            const outputs3 = layer2.forward(&outputs2.fwd_out);

            var max_guess: f64 = std.math.floatMin(f64);
            var guess_index: usize = 0;
            for (0..BATCH_SIZE) |bi| {
                for (outputs3.outputs[bi * OUTPUT_SIZE .. (bi + 1) * OUTPUT_SIZE], 0..) |o, oi| {
                    if (o > max_guess) {
                        max_guess = o;
                        guess_index = oi;
                    }
                }
                if (guess_index == mnist_data.test_labels[b + bi]) {
                    correct += 1;
                }
            }
        }
        correct = correct / 10000;
        std.debug.print("Average Validation Accuracy: {}\n", .{correct});
    }
}
fn averageArray(arr: []const f64) f64 {
    var sum: f64 = 0;
    for (arr) |elem| {
        sum += elem;
    }
    return sum / @as(f64, @floatFromInt(arr.len));
}

test "Forward once" {
    //var allocator = std.testing.allocator;
    const b = 2;
    const loss_function = nll.NLL(2, b);

    // Create layer with custom weights
    var layer1 = layer.Layer(2, 2, b).init();
    //allocator.free(layer1.weights);
    const custom_weights = [4]f64{ 0.1, 0.2, 0.3, 0.4 };
    layer1.weights = custom_weights;

    // Test forward pass outputs
    const inputs = [_]f64{ 0.1, 0.2, 0.3, 0.4 };
    const outputs = layer1.forward(&inputs);
    const expected_outputs = [4]f64{
        0.07,
        0.1,
        0.15,
        0.22,
    };
    var i: u32 = 0;

    //std.debug.print("  batch outputs: {any}\n", .{outputs});
    while (i < 4) : (i += 1) {
        // std.debug.print("output: {any} , expected: {any}\n", .{ outputs[i], expected_outputs[i] });
        try std.testing.expectApproxEqRel(expected_outputs[i], outputs[i], 0.000000001);
    }

    // Test loss outputs
    var targets_array = [_]u8{ 0, 1 };
    const targets: []u8 = &targets_array;
    const loss = loss_function.nll(outputs, targets);
    //allocator.free(outputs);
    const expected_loss = [2]f64{ 0.7082596763414484, 0.658759555548697 };
    i = 0;
    while (i < 2) : (i += 1) {
        try std.testing.expectApproxEqRel(loss.loss[i], expected_loss[i], 0.000000001);
    }

    // Test loss input_grads
    const expected_loss_input_grads = [4]f64{
        -5.074994375506203e-01,
        5.074994375506204e-01,
        4.8250714233361025e-01,
        -4.8250714233361025e-01,
    };
    i = 0;
    while (i < 4) : (i += 1) {
        try std.testing.expectApproxEqRel(loss.input_grads[i], expected_loss_input_grads[i], 0.000000001);
    }

    // Do layer backwards
    const grads = layer1.backwards(loss.input_grads);

    // Test layer weight grads
    const expected_layer_weight_grads = [4]f64{
        4.700109947251052e-02,
        -4.7001099472510514e-02,
        4.575148471166002e-02,
        -4.5751484711660004e-02,
    };
    i = 0;
    //std.debug.print("\n output: {any} ,\n expected: {any}\n", .{
    //    grads.weight_grads,
    //    expected_layer_weight_grads,
    //});
    while (i < 4) : (i += 1) {
        try std.testing.expectApproxEqRel(
            expected_layer_weight_grads[i],
            grads.weight_grads[i],
            0.000_000_001,
        );
    }

    // Test layer input grads
    const expected_layer_input_grads = [4]f64{
        5.074994375506206e-02,
        5.074994375506209e-02,
        -4.8250714233361025e-02,
        -4.8250714233361025e-02,
    };
    i = 0;

    //std.debug.print("output: {any} , expected: {any}\n", .{
    //    grads.input_grads,
    //    expected_layer_input_grads,
    //});

    while (i < 4) : (i += 1) {
        //std.debug.print("output: {any} , expected: {any}\n", .{
        //    grads.input_grads[i],
        //    expected_layer_input_grads[i],
        //});
        try std.testing.expectApproxEqRel(grads.input_grads[i], expected_layer_input_grads[i], 0.000000001);
    }
}

//test "Train Memory Leak" {
//    var allocator = std.testing.allocator;
//
//    // Get MNIST data
//    const mnist_data = try mnist.readMnist(allocator);
//
//    // Prep loss function
//    const loss_function = nll.NLL(OUTPUT_SIZE);
//
//    // Prep NN
//    var layer1 = try layer.Layer(INPUT_SIZE, 100).init(allocator);
//    var relu1 = relu.Relu.new();
//    var layer2 = try layer.Layer(100, OUTPUT_SIZE).init(allocator);
//
//    // Prep inputs and targets
//    const inputs = mnist_data.train_images[0..INPUT_SIZE];
//    const targets = mnist_data.train_labels[0..1];
//
//    // Go forward and get loss
//    const outputs1 = try layer1.forward(inputs, allocator);
//    const outputs2 = try relu1.forward(outputs1, allocator);
//    const outputs3 = try layer2.forward(outputs2, allocator);
//    const loss = try loss_function.nll(outputs3, targets, allocator);
//
//    // Update network
//    const grads1 = try layer2.backwards(loss.input_grads, allocator);
//    const grads2 = try relu1.backwards(grads1.input_grads, allocator);
//    const grads3 = try layer1.backwards(grads2, allocator);
//    layer1.applyGradients(grads3.weight_grads);
//    layer2.applyGradients(grads1.weight_grads);
//
//    // Free memory
//    allocator.free(outputs1);
//    allocator.free(outputs2);
//    allocator.free(outputs3);
//    grads1.deinit(allocator);
//    allocator.free(grads2);
//    grads3.deinit(allocator);
//    loss.deinit(allocator);
//
//    layer1.deinit(allocator);
//    layer2.deinit(allocator);
//    mnist_data.deinit(allocator);
//}
