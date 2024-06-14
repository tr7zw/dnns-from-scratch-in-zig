const layer = @import("layer.zig");
const layerB = @import("layerBias.zig");
const nll = @import("nll.zig");
const mnist = @import("mnist.zig");
const relu = @import("relu.zig");
const pyramid = @import("pyramid.zig");
const gaussian = @import("gaussian.zig");

const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    //const l = [_]usize{100};
    const NND = [_]layerDescriptor{ .{
        .layer = .{ .LayerB = 100 },
        .activation = Activation{ .gaussian = gaussian{} },
    }, .{
        .layer = .{ .LayerB = 10 },
        .activation = Activation{ .none = {} },
    } };
    _ = try Neuralnet(
        &NND,
        784,
        10,
        100,
        32,
        allocator,
    );
}

const Activation = union {
    relu: relu,
    pyramid: pyramid,
    gaussian: gaussian,
    none: void,
};

const uLayer = union(enum) {
    LayerB: usize,
    Layer: usize,
};
const Layer = union(enum) {
    LayerB: layerB,
    Layer: layer,
};

const layerDescriptor = struct {
    layer: uLayer,
    activation: Activation,
};

const layerStorage = struct {
    layer: Layer,
    activation: Activation,
};

fn layerFromDescriptor(alloc: std.mem.Allocator, comptime desc: layerDescriptor, batchSize: usize, inputSize: usize) layerStorage {
    const layerType = switch (desc.layer) {
        .Layer => |size| Layer{ .Layer = layer.init(
            alloc,
            batchSize,
            inputSize,
            size,
        ) },
        .LayerB => |size| Layer{ .LayerB = layerB.init(
            alloc,
            batchSize,
            inputSize,
            size,
        ) },
    };
    return .{
        .layer = layerType,
        .activation = try desc.activation.init(
            alloc,
            batchSize,
            layerType.outputSize,
        ),
    };
}

pub fn Neuralnet(
    comptime layers: []const layerDescriptor,
    comptime inputSize: u32,
    comptime outputSize: u32,
    comptime batchSize: u32,
    comptime epochs: u32,
    allocator: std.mem.Allocator,
) ![layers.len]layerStorage {

    //var descriptors: [layers.len]layerDescriptor = undefined;
    std.debug.assert(outputSize == switch (layers[layers.len - 1].layer) {
        .Layer, .LayerB => |l| l,
    });

    const Loss = nll.NLL(outputSize);

    const testImageCount = 10000;

    // Get MNIST data
    const mnist_data = try mnist.readMnist(allocator);
    defer mnist_data.deinit(allocator);

    comptime var previousLayerSize = inputSize;
    var storage: [layers.len]layerStorage = undefined;
    var validationStorage: [layers.len]layerStorage = undefined;
    // Prep NN

    inline for (layers, 0..) |layerD, i| {
        const size = switch (layerD.layer) {
            .Layer, .LayerB => |size| size,
        };
        defer previousLayerSize = size;
        storage[i] = layerFromDescriptor(
            allocator,
            layerD,
            batchSize,
            previousLayerSize,
        );
        //const b = if (layerD.activation) |b| try b.init(allocator, batchSize) else null;
        validationStorage[i] = layerFromDescriptor(
            allocator,
            layerD,
            testImageCount,
            previousLayerSize,
        );
    }

    var loss: Loss = try Loss.init(allocator, batchSize);

    const t = std.time.milliTimestamp();
    std.debug.print("Training... \n", .{});
    // Do training
    var e: usize = 0;
    while (e < epochs) : (e += 1) {
        // Do training
        var i: usize = 0;
        while (i < 60000 / batchSize) : (i += 1) {
            //if (i % (10000 / BATCH_SIZE) == 0) {
            //    const ct = std.time.milliTimestamp();
            //    std.debug.print("batch number: {}, time total: {}ms\n", .{ i * BATCH_SIZE, ct - t });
            //    //t = ct;
            //    std.debug.print("\n l2 bias:\n {any},\n", .{
            //        layer2.biases,
            //    });
            //}
            // Prep inputs and targets
            const inputs = mnist_data.train_images[i * inputSize * batchSize .. (i + 1) * inputSize * batchSize];
            const targets = mnist_data.train_labels[i * batchSize .. (i + 1) * batchSize];

            // Go forward and get loss
            //layer1.forward(inputs);
            //relu1.forward(layer1.outputs);
            //layer2.forward(relu1.fwd_out);

            var previousLayerOut = inputs;
            for (&storage) |current| {
                switch (current.layer) {
                    inline else => |currentlayer| {
                        currentlayer.forward(previousLayerOut);
                        previousLayerOut = currentlayer.outputs;
                        current.activation.forward(previousLayerOut);
                        previousLayerOut = current.activation.fwd_out;
                    },
                }
                //current.layer.forward(previousLayerOut);
                //previousLayerOut = current.layer.outputs;
                //current.activation.forward(previousLayerOut);
                //previousLayerOut = current.activation.fwd_out;
            }

            loss.nll(previousLayerOut, targets) catch |err| {
                const ct = std.time.milliTimestamp();
                std.debug.print("batch number: {}, time delta: {}ms\n", .{ i * batchSize, ct - t });
                std.debug.print("average loss for batch: {any}\n", .{
                    averageArray(loss.loss),
                });
                return err;
            };
            var previousGradient = loss.input_grads;
            for (storage.len..0) |current| {
                current.activation.backwards(previousGradient);
                previousGradient = current.activation.bkw_out;
                current.layer.backwards(previousGradient);
                previousGradient = current.layer.input_grads;
            }
            // Update network
            //layer2.backwards(loss.input_grads);
            //relu1.backwards(layer2.input_grads);
            //layer1.backwards(relu1.bkw_out);

            for (storage) |current| {
                current.layer.applyGradients();
            }
            //layer1.applyGradients();
            //layer2.applyGradients();
        }

        // Do validation
        i = 0;
        var correct: f64 = 0;
        var b: usize = 0;
        const inputs = mnist_data.test_images;

        for (validationStorage, 0..) |current, cur| {
            current.layer.setWeights(storage[cur].layer.weights);
            current.layer.setBiases(storage[cur].layer.biases);
        }
        var previousLayerOut = inputs;
        for (validationStorage) |current| {
            current.layer.forward(previousLayerOut);
            previousLayerOut = current.layer.outputs;
            current.activation.forward(previousLayerOut);
            previousLayerOut = current.activation.fwd_out;
        }

        //validationLayer1.setWeights(layer1.weights);
        //validationLayer2.setWeights(layer2.weights);
        //validationLayer1.setBiases(layer1.biases);
        //validationLayer2.setBiases(layer2.biases);

        //validationLayer1.forward(inputs);
        //validationRelu.forward(validationLayer1.outputs);
        //validationLayer2.forward(validationRelu.fwd_out);

        while (b < 10000) : (b += 1) {
            var max_guess: f64 = std.math.floatMin(f64);
            var guess_index: usize = 0;
            for (previousLayerOut[b * outputSize .. (b + 1) * outputSize], 0..) |o, oi| {
                if (o > max_guess) {
                    max_guess = o;
                    guess_index = oi;
                }
            }
            if (guess_index == mnist_data.test_labels[b]) {
                correct += 1;
            }
        }
        correct = correct / 10000;
        std.debug.print("Average Validation Accuracy: {}\n", .{correct});
    }
    const ct = std.time.milliTimestamp();
    std.debug.print(" time total: {}ms\n", .{ct - t});
    return storage;
}
fn averageArray(arr: []f64) f64 {
    var sum: f64 = 0;
    for (arr) |elem| {
        sum += elem;
    }
    return sum / @as(f64, @floatFromInt(arr.len));
}

test "Forward once" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var AA = std.heap.ArenaAllocator.init(gpa.allocator());
    const allocator = AA.allocator();
    defer _ = AA.deinit();

    const b = 2;
    var loss = try nll.NLL(2, b).init(allocator);

    // Create layer with custom weights
    var layer1 = try layer.Layer(2, 2, b).init(allocator);
    //allocator.free(layer1.weights);
    var custom_weights = [4]f64{ 0.1, 0.2, 0.3, 0.4 };
    //layer1.weights = custom_weights;
    layer1.setWeights(&custom_weights);

    // Test forward pass outputs
    const inputs = [_]f64{ 0.1, 0.2, 0.3, 0.4 };
    layer1.forward(&inputs);
    const outputs = layer1.outputs;
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
    try loss.nll(outputs, targets);
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
    layer1.backwards(loss.input_grads);

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
            layer1.weight_grads[i],
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
        try std.testing.expectApproxEqRel(layer1.input_grads[i], expected_layer_input_grads[i], 0.000000001);
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
