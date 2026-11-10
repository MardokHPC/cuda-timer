#include <gtest/gtest.h>

// Host-only main: this binary runs on GPU-less CI runners because its suite
// (loggers, export, Stats, types) never touches a device. Device suites link
// test_main_device.cpp, which exits 77 when no device is present.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
