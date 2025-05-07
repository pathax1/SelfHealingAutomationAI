Feature: IKEA Logout Simulation

  Scenario: Simulate logout by navigating to homepage
    Given the user is on the login page
    When the user clicks on the IKEA logo
    Then the user should be redirected to the homepage
