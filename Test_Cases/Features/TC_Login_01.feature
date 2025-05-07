Feature: ZARA Login Page Accessibility

  Scenario: Navigate to ZARA login page
    Given the user has launched the ZARA site and rejected cookies
    When the user clicks on the login link
    Then the login form should be displayed
