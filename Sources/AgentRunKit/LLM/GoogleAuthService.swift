import Foundation

// MARK: - ADC Credential File

private struct ADCCredentials: Decodable {
    let type: String
    let clientId: String
    let clientSecret: String
    let refreshToken: String

    enum CodingKeys: String, CodingKey {
        case type
        case clientId = "client_id"
        case clientSecret = "client_secret"
        case refreshToken = "refresh_token"
    }
}

// MARK: - Token Response

private struct TokenResponse: Decodable {
    let accessToken: String
    let expiresIn: Int
    let tokenType: String

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case expiresIn = "expires_in"
        case tokenType = "token_type"
    }
}

/// Manages Google OAuth2 tokens from Application Default Credentials (ADC).
///
/// Reads `~/.config/gcloud/application_default_credentials.json` (created by
/// `gcloud auth application-default login`) and transparently refreshes access
/// tokens as needed.
///
/// Thread-safe via `actor` isolation — only one refresh request can be in
/// flight at a time.
public actor GoogleAuthService {
    // MARK: - Errors

    public enum GoogleAuthError: Error, LocalizedError, Sendable {
        case credentialsFileNotFound(path: String)
        case unsupportedCredentialType(String)
        case refreshFailed(statusCode: Int, body: String)
        case decodingFailed(String)

        public var errorDescription: String? {
            switch self {
            case let .credentialsFileNotFound(path):
                "Google ADC credentials not found at \(path). Run `gcloud auth application-default login`."
            case let .unsupportedCredentialType(type):
                "Unsupported ADC credential type: \(type). Only 'authorized_user' is supported."
            case let .refreshFailed(code, body):
                "Token refresh failed (HTTP \(code)): \(body)"
            case let .decodingFailed(message):
                "Failed to decode ADC credentials: \(message)"
            }
        }
    }

    // MARK: - State

    private let clientID: String
    private let clientSecret: String
    private let refreshToken: String
    private let session: URLSession

    private var cachedAccessToken: String?
    private var tokenExpiry: Date?

    /// Refresh the token when it has fewer than this many seconds remaining.
    private let refreshMargin: TimeInterval = 300 // 5 minutes

    private static let tokenEndpoint = URL(string: "https://oauth2.googleapis.com/token")!

    // MARK: - Init

    /// Creates an auth service by reading the ADC file at the default path.
    public init(session: URLSession = .shared) throws {
        try self.init(credentialsPath: Self.defaultCredentialsPath(), session: session)
    }

    /// Creates an auth service by reading the ADC file at a custom path.
    public init(credentialsPath: String, session: URLSession = .shared) throws {
        guard FileManager.default.fileExists(atPath: credentialsPath) else {
            throw GoogleAuthError.credentialsFileNotFound(path: credentialsPath)
        }
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: credentialsPath))
        } catch {
            throw GoogleAuthError.decodingFailed("Failed to read file: \(error.localizedDescription)")
        }
        let credentials: ADCCredentials
        do {
            credentials = try JSONDecoder().decode(ADCCredentials.self, from: data)
        } catch {
            throw GoogleAuthError.decodingFailed(error.localizedDescription)
        }
        guard credentials.type == "authorized_user" else {
            throw GoogleAuthError.unsupportedCredentialType(credentials.type)
        }
        clientID = credentials.clientId
        clientSecret = credentials.clientSecret
        refreshToken = credentials.refreshToken
        self.session = session
    }

    // MARK: - Public API

    /// Returns a valid access token, refreshing if necessary.
    public func accessToken() async throws -> String {
        if let token = cachedAccessToken,
           let expiry = tokenExpiry,
           Date() < expiry.addingTimeInterval(-refreshMargin) {
            return token
        }
        return try await refreshAccessToken()
    }

    // MARK: - Private

    private func refreshAccessToken() async throws -> String {
        var request = URLRequest(url: Self.tokenEndpoint)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")

        let body = [
            "client_id=\(urlEncode(clientID))",
            "client_secret=\(urlEncode(clientSecret))",
            "refresh_token=\(urlEncode(refreshToken))",
            "grant_type=refresh_token",
        ].joined(separator: "&")
        request.httpBody = Data(body.utf8)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw GoogleAuthError.refreshFailed(statusCode: 0, body: "Invalid response")
        }
        guard httpResponse.statusCode == 200 else {
            let responseBody = String(data: data, encoding: .utf8) ?? "<unreadable>"
            throw GoogleAuthError.refreshFailed(statusCode: httpResponse.statusCode, body: responseBody)
        }

        let tokenResponse = try JSONDecoder().decode(TokenResponse.self, from: data)
        cachedAccessToken = tokenResponse.accessToken
        tokenExpiry = Date().addingTimeInterval(TimeInterval(tokenResponse.expiresIn))
        return tokenResponse.accessToken
    }

    private func urlEncode(_ string: String) -> String {
        string.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? string
    }

    /// The default path to the ADC credentials file.
    public static func defaultCredentialsPath() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/.config/gcloud/application_default_credentials.json"
    }

    /// Whether an ADC credentials file exists at the default path.
    public static func credentialsAvailable() -> Bool {
        FileManager.default.fileExists(atPath: defaultCredentialsPath())
    }
}
