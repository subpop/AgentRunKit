@testable import AgentRunKit
import Foundation
import Testing

struct GoogleAuthServiceTests {
    @Test
    func defaultCredentialsPathFormat() {
        let path = GoogleAuthService.defaultCredentialsPath()
        #expect(path.hasSuffix("/.config/gcloud/application_default_credentials.json"))
        #expect(path.hasPrefix("/"))
    }

    @Test
    func initWithMissingFileThrows() {
        #expect(throws: GoogleAuthService.GoogleAuthError.self) {
            _ = try GoogleAuthService(credentialsPath: "/nonexistent/path/credentials.json")
        }
    }

    @Test
    func initWithMissingFileThrowsCorrectError() {
        do {
            _ = try GoogleAuthService(credentialsPath: "/tmp/does_not_exist_adc.json")
            Issue.record("Expected error")
        } catch let error as GoogleAuthService.GoogleAuthError {
            if case let .credentialsFileNotFound(path) = error {
                #expect(path == "/tmp/does_not_exist_adc.json")
            } else {
                Issue.record("Expected credentialsFileNotFound, got \(error)")
            }
        } catch {
            Issue.record("Expected GoogleAuthError, got \(error)")
        }
    }

    @Test
    func initWithInvalidJSONThrows() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("invalid_adc_\(UUID().uuidString).json")
        try Data("not json".utf8).write(to: tempFile)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        do {
            _ = try GoogleAuthService(credentialsPath: tempFile.path)
            Issue.record("Expected error")
        } catch let error as GoogleAuthService.GoogleAuthError {
            guard case .decodingFailed = error else {
                Issue.record("Expected decodingFailed, got \(error)")
                return
            }
        } catch {
            Issue.record("Expected GoogleAuthError, got \(error)")
        }
    }

    @Test
    func initWithUnsupportedTypeThrows() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("sa_adc_\(UUID().uuidString).json")
        let json = """
        {
            "type": "service_account",
            "client_id": "123",
            "client_secret": "secret",
            "refresh_token": "token"
        }
        """
        try Data(json.utf8).write(to: tempFile)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        do {
            _ = try GoogleAuthService(credentialsPath: tempFile.path)
            Issue.record("Expected error")
        } catch let error as GoogleAuthService.GoogleAuthError {
            guard case let .unsupportedCredentialType(type) = error else {
                Issue.record("Expected unsupportedCredentialType, got \(error)")
                return
            }
            #expect(type == "service_account")
        } catch {
            Issue.record("Expected GoogleAuthError, got \(error)")
        }
    }

    @Test
    func initWithValidCredentialsSucceeds() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("valid_adc_\(UUID().uuidString).json")
        let json = """
        {
            "type": "authorized_user",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "refresh_token": "test-refresh-token"
        }
        """
        try Data(json.utf8).write(to: tempFile)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        // Should not throw
        _ = try GoogleAuthService(credentialsPath: tempFile.path)
    }

    @Test
    func credentialsAvailableReturnsConsistentResult() {
        let available = GoogleAuthService.credentialsAvailable()
        let path = GoogleAuthService.defaultCredentialsPath()
        let fileExists = FileManager.default.fileExists(atPath: path)
        #expect(available == fileExists)
    }

    @Test
    func errorDescriptionsAreNonEmpty() throws {
        let errors: [GoogleAuthService.GoogleAuthError] = [
            .credentialsFileNotFound(path: "/test"),
            .unsupportedCredentialType("service_account"),
            .refreshFailed(statusCode: 401, body: "Unauthorized"),
            .decodingFailed("test error")
        ]
        for error in errors {
            let description = try #require(error.errorDescription)
            #expect(!description.isEmpty)
        }
    }
}
