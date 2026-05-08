import SwiftUI

@main
struct KokoroAppApp: App {
    var body: some Scene {
        WindowGroup {
            if MemoryTestRunner.isRequested {
                MemoryTestView()
            } else {
                ContentView()
                    .preferredColorScheme(.dark)
            }
        }
    }
}
